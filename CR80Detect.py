import math
import cv2
import numpy as np

CARD_RATIO = 85.60 / 53.98  # CR80 aspect ratio ~1.586
CR80_LONG_SIDE_MM = 85.60


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


def order_corners(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    sums = pts[:, 0] + pts[:, 1]
    tl_idx = np.argmin(sums)
    pts = np.roll(pts, -tl_idx, axis=0)
    if pts[1, 0] < pts[3, 0]:
        pts = pts[[0, 3, 2, 1]]
    return pts


def get_perspective_flattened_long_side(quad):
    """
    Computes the true unwarped/perspective-flattened long side of a CR80 card.
    Accounts for projective foreshortening (pitch and yaw tilt) using the harmonic
    mean of opposing edges and the known CR80 aspect ratio (1.5858).
    """
    if quad is None:
        return None
    pts = order_corners(quad)
    tl, tr, br, bl = pts

    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)

    # Harmonic mean gives perspective-invariant length at card center depth
    w_proj = 2.0 * w_top * w_bot / max(1e-6, w_top + w_bot)
    h_proj = 2.0 * h_left * h_right / max(1e-6, h_left + h_right)

    if w_proj >= h_proj:
        # Card is horizontal: width is long side
        observed_ratio = w_proj / max(1e-6, h_proj)
        if observed_ratio >= CARD_RATIO:
            # Pitch tilt (forehead slope): height is foreshortened, width is true scale
            return float(w_proj)
        else:
            # Yaw tilt: width is foreshortened, height * ratio gives true frontal width
            return float(h_proj * CARD_RATIO)
    else:
        # Card is vertical: height is long side
        observed_ratio = h_proj / max(1e-6, w_proj)
        if observed_ratio >= CARD_RATIO:
            return float(h_proj)
        else:
            return float(w_proj * CARD_RATIO)


def find_contour_quads(frame_edges, min_area=1500):
    contours, _ = cv2.findContours(frame_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.035 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quads.append(approx.reshape(4, 2).astype(np.int32))
    return quads


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


class CardTracker:
    def __init__(self, alpha=0.35, max_missing_frames=5):
        self.alpha = alpha
        self.max_missing = max_missing_frames
        self.missing_count = 0
        self.smoothed_quad = None

    def update(self, detected_quad):
        if detected_quad is not None:
            ordered = order_corners(detected_quad)
            if self.smoothed_quad is None:
                self.smoothed_quad = ordered
            else:
                self.smoothed_quad = self.alpha * ordered + (1.0 - self.alpha) * self.smoothed_quad
            self.missing_count = 0
            return self.smoothed_quad.astype(np.int32)
        else:
            if self.smoothed_quad is not None and self.missing_count < self.max_missing:
                self.missing_count += 1
                return self.smoothed_quad.astype(np.int32)
            else:
                self.smoothed_quad = None
                return None


_card_tracker = CardTracker(alpha=0.35, max_missing_frames=5)


def detect_cr80(frame_edges, theta_div=549, hough_thresh=56, min_length=30, max_gap=29, angle_tol=7, corner_slack=57, overlap_pct=0.0, infer_4th=True):
    # 1. Closed contour quad candidates (stable, unbroken edges)
    contour_quads = find_contour_quads(frame_edges, min_area=1500)

    # 2. Hough line quad candidates
    lines = cv2.HoughLinesP(frame_edges, 1, np.pi / theta_div, threshold=hough_thresh, minLineLength=min_length, maxLineGap=max_gap)
    hough_quads = find_candidate_quads(lines, min_len=min_length, angle_tol=angle_tol, corner_slack=corner_slack, infer_4th=infer_4th)

    # Combine candidates and remove duplicates
    all_quads = contour_quads + hough_quads
    quads = suppress_overlapping_quads(all_quads, overlap_thresh=overlap_pct)

    raw_best = None
    if quads:
        scored = sorted(((quad_score(q), q) for q in quads), key=lambda x: x[0])
        best_score, raw_best = scored[0]

    # Temporal smoothing and persistence to eliminate frame flickering
    stable_best = _card_tracker.update(raw_best)

    return lines, quads, stable_best
