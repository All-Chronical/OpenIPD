import cv2
import math
import numpy as np

CARD_RATIO = 85.60 / 53.98  # CR80 aspect ratio ~1.586


def line_intersect(s1, s2):
    p, r = s1[:2], s1[2:] - s1[:2]
    q, s = s2[:2], s2[2:] - s2[:2]
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-9:
        return None
    t = ((q - p)[0] * s[1] - (q - p)[1] * s[0]) / den
    return p + t * r


def on_segment(c, s, slack=60.0):
    p, d = s[:2], s[2:] - s[:2]
    L = math.hypot(d[0], d[1])
    if L < 1e-6:
        return False
    t = ((c - p) @ d) / (L * L)
    return -slack / L <= t <= 1.0 + slack / L


def find_candidate_quads(lines, min_len=40, angle_tol=7, min_gap=30, max_gap=400, corner_slack=60.0, infer_4th=False):
    if lines is None:
        return []
    seg = lines.reshape(-1, 4).astype(float)
    length = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    seg = seg[length >= min_len]
    if len(seg) == 0:
        return []
    angle = np.degrees(np.arctan2(seg[:, 3] - seg[:, 1], seg[:, 2] - seg[:, 0])) % 180
    mid = (seg[:, :2] + seg[:, 2:]) / 2

    quads = []
    for i in range(len(seg)):
        ai = math.radians(angle[i])
        u = np.array([math.cos(ai), math.sin(ai)])
        n = np.array([-u[1], u[0]])
        for j in range(i + 1, len(seg)):
            if abs(((angle[i] - angle[j] + 90) % 180) - 90) > angle_tol:
                continue
            gap = abs(n @ (mid[j] - mid[i]))
            if not (min_gap <= gap <= max_gap):
                continue
            orth = (angle[i] + 90) % 180
            lo, hi = sorted([n @ mid[i], n @ mid[j]])
            cross = [k for k in range(len(seg))
                     if abs(((angle[k] - orth + 90) % 180) - 90) <= angle_tol
                     and lo < n @ mid[k] < hi]
            if len(cross) >= 2:
                left = min(cross, key=lambda k: u @ mid[k])
                right = max(cross, key=lambda k: u @ mid[k])
                corners = [line_intersect(seg[i], seg[left]), line_intersect(seg[i], seg[right]),
                           line_intersect(seg[j], seg[right]), line_intersect(seg[j], seg[left])]
                if any(c is None for c in corners):
                    continue
                if not (all(on_segment(c, seg[i], corner_slack) for c in (corners[0], corners[1]))
                        and all(on_segment(c, seg[j], corner_slack) for c in (corners[2], corners[3]))
                        and all(on_segment(c, seg[left], corner_slack) for c in (corners[0], corners[3]))
                        and all(on_segment(c, seg[right], corner_slack) for c in (corners[1], corners[2]))):
                    continue
                quads.append(np.array(corners, dtype=np.int32))
            elif len(cross) == 1 and infer_4th:
                known = cross[0]
                for d in (1.586 * gap, gap / 1.586):
                    for sign in (+1, -1):
                        p0 = mid[known] + sign * d * u
                        inferred = np.array([p0 - 100 * n, p0 + 100 * n]).ravel()
                        corners = [line_intersect(seg[i], seg[known]), line_intersect(seg[i], inferred),
                                   line_intersect(seg[j], inferred), line_intersect(seg[j], seg[known])]
                        if any(c is None for c in corners):
                            continue
                        if not (all(on_segment(c, seg[i], corner_slack) for c in (corners[0], corners[1]))
                                and all(on_segment(c, seg[j], corner_slack) for c in (corners[2], corners[3]))
                                and all(on_segment(c, seg[known], corner_slack) for c in (corners[0], corners[3]))):
                            continue
                        quads.append(np.array(corners, dtype=np.int32))
    return quads


def quad_score(quad):
    edges = [quad[(i + 1) % 4] - quad[i] for i in range(4)]
    lengths = [math.hypot(e[0], e[1]) for e in edges]
    side_a = (lengths[0] + lengths[2]) / 2.0
    side_b = (lengths[1] + lengths[3]) / 2.0
    if side_a < 1e-6 or side_b < 1e-6:
        return float("inf")
    ratio = max(side_a, side_b) / min(side_a, side_b)
    return abs(ratio - CARD_RATIO)


def suppress_overlapping_quads(quads, overlap_thresh=0.5):
    if len(quads) <= 1:
        return quads

    sorted_quads = sorted(quads, key=cv2.contourArea, reverse=True)
    kept = []
    boxes = [cv2.boundingRect(q) for q in sorted_quads]

    for i, q in enumerate(sorted_quads):
        x1, y1, w1, h1 = boxes[i]
        area1 = w1 * h1
        if area1 == 0:
            continue

        duplicate = False
        for k_idx in range(len(kept)):
            x2, y2, w2, h2 = boxes[k_idx]
            area2 = w2 * h2

            xi1 = max(x1, x2)
            yi1 = max(y1, y2)
            xi2 = min(x1 + w1, x2 + w2)
            yi2 = min(y1 + h1, y2 + h2)

            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            iou = inter / float(area1 + area2 - inter)

            if iou > overlap_thresh:
                duplicate = True
                break

        if not duplicate:
            kept.append(q)

    return kept


WINDOW_NAME = "OpenIPD Alpha"
cv2.namedWindow(WINDOW_NAME)


def nothing(x):
    pass


cv2.createTrackbar("Freeze (0/1)", WINDOW_NAME, 0, 1, nothing)
cv2.createTrackbar("Blur", WINDOW_NAME, 8, 15, nothing)
cv2.createTrackbar("Canny Low", WINDOW_NAME, 55, 255, nothing)
cv2.createTrackbar("Canny High", WINDOW_NAME, 82, 255, nothing)
cv2.createTrackbar("Theta (pi/X)", WINDOW_NAME, 549, 720, nothing)
cv2.createTrackbar("Hough Thresh", WINDOW_NAME, 56, 200, nothing)
cv2.createTrackbar("Min Length", WINDOW_NAME, 30, 200, nothing)
cv2.createTrackbar("Max Gap", WINDOW_NAME, 29, 150, nothing)
cv2.createTrackbar("Angle Tol", WINDOW_NAME, 7, 30, nothing)
cv2.createTrackbar("Corner Slack", WINDOW_NAME, 57, 150, nothing)
cv2.createTrackbar("Overlap (%)", WINDOW_NAME, 0, 100, nothing)
cv2.createTrackbar("Infer 4th (0/1)", WINDOW_NAME, 1, 1, nothing)

camera = cv2.VideoCapture(0)
cached_frame = None

while True:
    freeze = cv2.getTrackbarPos("Freeze (0/1)", WINDOW_NAME)
    if not freeze or cached_frame is None:
        ret, frame_read = camera.read()
        if not ret:
            break
        cached_frame = frame_read.copy()

    frame = cached_frame.copy()

    blur_val = cv2.getTrackbarPos("Blur", WINDOW_NAME)
    k = max(1, blur_val if blur_val % 2 == 1 else blur_val + 1)

    canny_low = cv2.getTrackbarPos("Canny Low", WINDOW_NAME)
    canny_high = max(canny_low + 1, cv2.getTrackbarPos("Canny High", WINDOW_NAME))

    theta_div = max(1, cv2.getTrackbarPos("Theta (pi/X)", WINDOW_NAME))
    hough_thresh = max(1, cv2.getTrackbarPos("Hough Thresh", WINDOW_NAME))
    min_length = max(1, cv2.getTrackbarPos("Min Length", WINDOW_NAME))
    max_gap = cv2.getTrackbarPos("Max Gap", WINDOW_NAME)

    angle_tol = max(1, cv2.getTrackbarPos("Angle Tol", WINDOW_NAME))
    corner_slack = cv2.getTrackbarPos("Corner Slack", WINDOW_NAME)
    overlap_pct = max(0, cv2.getTrackbarPos("Overlap (%)", WINDOW_NAME)) / 100.0
    infer_4th = bool(cv2.getTrackbarPos("Infer 4th (0/1)", WINDOW_NAME))

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray, (k, k), 0)
    frame_edges = cv2.Canny(frame_blur, canny_low, canny_high)

    lines = cv2.HoughLinesP(frame_edges, 1, np.pi / theta_div, threshold=hough_thresh, minLineLength=min_length, maxLineGap=max_gap)
    quads = find_candidate_quads(lines, min_len=min_length, angle_tol=angle_tol, corner_slack=corner_slack, infer_4th=infer_4th)
    quads = suppress_overlapping_quads(quads, overlap_thresh=overlap_pct)

    frame = cv2.cvtColor(frame_edges, cv2.COLOR_GRAY2BGR)

    # 1. Raw Hough lines: red
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # 2. Candidate quads: filled blue + blue outline
    if quads:
        overlay = frame.copy()
        for q in quads:
            cv2.fillPoly(overlay, [q.reshape(-1, 1, 2)], (255, 0, 0))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        for q in quads:
            cv2.drawContours(frame, [q.reshape(-1, 1, 2)], 0, (255, 0, 0), 2)

    # 3. Winner: green
    if quads:
        scored = sorted(((quad_score(q), q) for q in quads), key=lambda x: x[0])
        best_score, best_quad = scored[0]
        cv2.drawContours(frame, [best_quad.reshape(-1, 1, 2)], 0, (0, 255, 0), 3)

    if freeze:
        cv2.putText(frame, "FROZEN (Press 'F' or Space to toggle)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("OpenIPD Alpha", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key in (ord('f'), ord('F'), 32):
        curr = cv2.getTrackbarPos("Freeze (0/1)", WINDOW_NAME)
        cv2.setTrackbarPos("Freeze (0/1)", WINDOW_NAME, 0 if curr else 1)

