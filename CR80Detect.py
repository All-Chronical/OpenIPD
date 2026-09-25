import math
import cv2
import numpy as np

CARD_RATIO = 85.60 / 53.98  # CR80 aspect ratio ~1.58577
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


def make_strict_cr80_box(pts):
    """
    Fits an oriented rectangle to the given points and strictly enforces
    the exact CR80 reference ratio (85.60 / 53.98). Only scales, translates,
    and rotates based on the tracking data.
    """
    rect = cv2.minAreaRect(pts.astype(np.float32))
    (cx, cy), (w, h), angle = rect
    if w < h:
        w, h = h, w
        angle += 90.0

    r = w / max(1e-6, h)
    # Pitch foreshortening compresses height, yaw foreshortening compresses width
    if r >= CARD_RATIO:
        best_L = w
    else:
        best_L = h * CARD_RATIO

    best_H = best_L / CARD_RATIO
    box = cv2.boxPoints(((cx, cy), (best_L, best_H), angle)).astype(np.int32)
    return box, float(best_L), float(abs(r - CARD_RATIO))


def get_perspective_flattened_long_side(quad):
    """
    Extracts the perspective-normalized long side (in pixels) of the fitted CR80 card.
    """
    if quad is None:
        return None
    rect = cv2.minAreaRect(quad.astype(np.float32))
    (cx, cy), (w, h), angle = rect
    if w < h:
        w, h = h, w
    r = w / max(1e-6, h)
    if r >= CARD_RATIO:
        return float(w)
    else:
        return float(h * CARD_RATIO)


def compute_calibrated_ipd(pupil_dist, card_len, depth_offset=0.0):
    """
    Computes IPD in mm calibrated for the anterior depth offset of the forehead.
    Because the card on the forehead is closer to the camera than the retinas/pupils,
    geometric perspective slightly magnifies the card relative to the eyes.
    Using MediaPipe's 3D depth_offset recovers the true IPD accurately.
    """
    if pupil_dist is None or card_len is None or card_len < 1e-3:
        return None
    raw_ipd = (pupil_dist / card_len) * CR80_LONG_SIDE_MM
    calibrated_ipd = raw_ipd * (1.0 + 0.35 * max(0.0, float(depth_offset)))
    return calibrated_ipd


def score_card_candidate(strict_box, L, ratio_err, pupil_dist=None, frame_gray=None, frame_edges=None, depth_offset=0.0, mid_pupil_x=None, roi=None):
    """
    Scores candidates to ensure the full outer card boundary is chosen over
    internal barcode text, hair loops, or the outer face contour.
    Does not assume card color or brightness (supports white, dark red, blue, black, etc.).
    """
    area = cv2.contourArea(strict_box)
    if area < 2000:
        return float("inf")

    # If pupils are available, enforce physical scale band and forehead positioning
    if pupil_dist is not None and pupil_dist > 10:
        ratio_to_pupils = L / pupil_dist
        if ratio_to_pupils < 1.25 or ratio_to_pupils > 1.65:
            return float("inf")

        cx, cy = np.mean(strict_box, axis=0)

        # Forehead geometric constraints: card must be horizontally aligned with face
        if mid_pupil_x is not None:
            if abs(cx - mid_pupil_x) > 0.65 * pupil_dist:
                return float("inf")

        if roi is not None:
            rx1, ry1, rx2, ry2 = roi
            if not (rx1 - 35 <= cx <= rx2 + 35 and ry1 - 40 <= cy <= ry2 + 40):
                return float("inf")

        # Edge support verification: check density of Canny edges along perimeter
        density = 0.0
        if frame_edges is not None:
            edge_mask = np.zeros_like(frame_edges)
            cv2.polylines(edge_mask, [strict_box], isClosed=True, color=255, thickness=3)
            matching_edges = np.sum((frame_edges > 0) & (edge_mask > 0))
            density = matching_edges / max(1.0, cv2.arcLength(strict_box, True))
            if density < 0.15:  # Must have at least 15% edge confirmation
                return float("inf")

        # Nominal expected card length accounting for physical CR80/IPD ratio and depth
        expected_L = (CR80_LONG_SIDE_MM / 60.50) * pupil_dist * (1.0 + 0.35 * max(0.0, float(depth_offset)))
        scale_err = abs(L - expected_L) / expected_L
        return scale_err * 2.5 + ratio_err * 0.4 - density * 0.4

    return ratio_err / (area ** 0.4)


def find_contour_quads(frame_edges, pupil_dist=None, min_area=2500, frame_gray=None, depth_offset=0.0, mid_pupil_x=None, roi=None):
    """
    Finds closed contours in the edge frame using RETR_LIST to discover
    internal card loops, constructing strictly proportioned CR80 candidate boxes.
    """
    contours, _ = cv2.findContours(frame_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        if w < h:
            w, h = h, w
        r = w / max(1e-6, h)

        # Ratio band check (allow for tilt)
        if 1.15 <= r <= 2.25:
            box, L, r_err = make_strict_cr80_box(cnt)
            score = score_card_candidate(
                box, L, r_err, pupil_dist=pupil_dist, frame_gray=frame_gray, frame_edges=frame_edges, depth_offset=depth_offset, mid_pupil_x=mid_pupil_x, roi=roi
            )
            if score < float("inf"):
                candidates.append((score, box))

    return candidates


def find_candidate_quads(lines, min_len=25, angle_tol=7, min_gap=50, max_gap=400, corner_slack=150.0, infer_4th=True, pupil_dist=None):
    if lines is None:
        return []
    seg = lines.reshape(-1, 4).astype(float)
    length = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    seg = seg[length >= min_len]
    if len(seg) == 0:
        return []
    angle = np.degrees(np.arctan2(seg[:, 3] - seg[:, 1], seg[:, 2] - seg[:, 0])) % 180
    mid = (seg[:, :2] + seg[:, 2:]) / 2

    quads_4sided = []
    quads_inferred = []

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

            # Check if opposing cross segments exist (4-sided quad)
            has_4sided = False
            if len(cross) >= 2:
                for k1 in range(len(cross)):
                    for k2 in range(k1 + 1, len(cross)):
                        c1, c2 = cross[k1], cross[k2]
                        span = abs(u @ (mid[c1] - mid[c2]))
                        if 1.2 * gap <= span <= 2.2 * gap:
                            c_left = c1 if (u @ mid[c1] < u @ mid[c2]) else c2
                            c_right = c2 if (c_left == c1) else c1
                            corners = [line_intersect(seg[i], seg[c_left]), line_intersect(seg[i], seg[c_right]),
                                       line_intersect(seg[j], seg[c_right]), line_intersect(seg[j], seg[c_left])]
                            if not any(c is None for c in corners):
                                quads_4sided.append(np.array(corners, dtype=np.int32))
                                has_4sided = True

            # 3-sided card detection: one side occluded by user fingers
            if not has_4sided and len(cross) >= 1 and infer_4th:
                u_pts = [u @ seg[i, :2], u @ seg[i, 2:], u @ seg[j, :2], u @ seg[j, 2:]]
                min_u, max_u = min(u_pts), max(u_pts)
                for known in cross:
                    u_k = u @ mid[known]
                    candidate_ds = [CARD_RATIO * gap]
                    # Check extent of detected parallel lines (where horizontal lines terminate)
                    d_max = max_u - u_k
                    if 1.35 * gap <= d_max <= 2.0 * gap:
                        candidate_ds.append(d_max)
                    d_min = u_k - min_u
                    if 1.35 * gap <= d_min <= 2.0 * gap:
                        candidate_ds.append(d_min)
                    if pupil_dist is not None:
                        candidate_ds.append(1.42 * pupil_dist)

                    for d in candidate_ds:
                        for sign in (+1, -1):
                            p0 = mid[known] + sign * d * u
                            inferred = np.array([p0 - 100 * n, p0 + 100 * n]).ravel()
                            corners = [line_intersect(seg[i], seg[known]), line_intersect(seg[i], inferred),
                                       line_intersect(seg[j], inferred), line_intersect(seg[j], seg[known])]
                            if not any(c is None for c in corners):
                                quads_inferred.append(np.array(corners, dtype=np.int32))

    return quads_4sided + quads_inferred


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


class ParametricCardTracker:
    """
    Smooths card position, scale, and angle independently across frames.
    Guarantees the resulting polygon is ALWAYS strictly in the 85.60 / 53.98
    CR80 reference aspect ratio without warping or deforming.
    """
    def __init__(self, alpha_pos=0.35, alpha_scale=0.20, alpha_rot=0.30, max_missing=6):
        self.alpha_pos = alpha_pos
        self.alpha_scale = alpha_scale
        self.alpha_rot = alpha_rot
        self.max_missing = max_missing
        self.missing_count = 0
        self.cx = None
        self.cy = None
        self.L = None
        self.angle = None

    def update(self, detected_quad):
        if detected_quad is not None:
            rect = cv2.minAreaRect(detected_quad.astype(np.float32))
            (raw_cx, raw_cy), (raw_w, raw_h), raw_angle = rect
            if raw_w < raw_h:
                raw_w, raw_h = raw_h, raw_w
                raw_angle += 90.0

            r = raw_w / max(1e-6, raw_h)
            raw_L = raw_w if r >= CARD_RATIO else raw_h * CARD_RATIO

            if self.L is None:
                self.cx, self.cy = raw_cx, raw_cy
                self.L = raw_L
                self.angle = raw_angle
            else:
                self.cx = self.alpha_pos * raw_cx + (1.0 - self.alpha_pos) * self.cx
                self.cy = self.alpha_pos * raw_cy + (1.0 - self.alpha_pos) * self.cy
                self.L = self.alpha_scale * raw_L + (1.0 - self.alpha_scale) * self.L
                diff_angle = (raw_angle - self.angle + 90) % 180 - 90
                self.angle += self.alpha_rot * diff_angle

            self.missing_count = 0
            H = self.L / CARD_RATIO
            box = cv2.boxPoints(((self.cx, self.cy), (self.L, H), self.angle)).astype(np.int32)
            return box
        else:
            if self.L is not None and self.missing_count < self.max_missing:
                self.missing_count += 1
                H = self.L / CARD_RATIO
                box = cv2.boxPoints(((self.cx, self.cy), (self.L, H), self.angle)).astype(np.int32)
                return box
            self.L = None
            return None


_card_tracker = ParametricCardTracker(alpha_pos=0.35, alpha_scale=0.20, alpha_rot=0.30, max_missing=6)


def detect_cr80(frame_edges, theta_div=549, hough_thresh=42, min_length=20, max_gap=25, angle_tol=7, corner_slack=150, overlap_pct=0.0, infer_4th=True, pupil_dist=None, frame_gray=None, depth_offset=0.0, mid_pupil_x=None, roi=None):
    # 1. Closed contour candidates in ROI using RETR_LIST
    contour_candidates = find_contour_quads(frame_edges, pupil_dist=pupil_dist, min_area=2500, frame_gray=frame_gray, depth_offset=depth_offset, mid_pupil_x=mid_pupil_x, roi=roi)

    # 2. Hough line candidates
    lines = cv2.HoughLinesP(frame_edges, 1, np.pi / theta_div, threshold=hough_thresh, minLineLength=min_length, maxLineGap=max_gap)
    hough_raw_quads = find_candidate_quads(lines, min_len=min_length, angle_tol=angle_tol, min_gap=45, corner_slack=corner_slack, infer_4th=infer_4th, pupil_dist=pupil_dist)

    hough_candidates = []
    for hq in hough_raw_quads:
        strict_hq, L, r_err = make_strict_cr80_box(hq)
        score = score_card_candidate(
            strict_hq, L, r_err, pupil_dist=pupil_dist, frame_gray=frame_gray, frame_edges=frame_edges, depth_offset=depth_offset, mid_pupil_x=mid_pupil_x, roi=roi
        )
        if score < float("inf"):
            hough_candidates.append((score, strict_hq))

    all_scored = contour_candidates + hough_candidates
    all_quads = [item[1] for item in all_scored]
    quads = suppress_overlapping_quads(all_quads, overlap_thresh=overlap_pct)

    raw_best = None
    if all_scored:
        sorted_scored = sorted(all_scored, key=lambda x: x[0])
        raw_best = sorted_scored[0][1]

    # Parametric smoothing: locks the aspect ratio to 85.60 / 53.98 while smoothly tracking position, scale, and rotation
    stable_best = _card_tracker.update(raw_best)

    return lines, quads, stable_best
