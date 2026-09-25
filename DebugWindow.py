import math
import cv2
import numpy as np
import CR80Detect

FEED_WINDOW = "OpenIPD Alpha"
CONTROLS_WINDOW = "Controls"

_smoothed_ipd = None


def get_smoothed_ipd(raw_ipd, alpha=0.25):
    global _smoothed_ipd
    if raw_ipd is None:
        return None
    if _smoothed_ipd is None:
        _smoothed_ipd = raw_ipd
    else:
        _smoothed_ipd = alpha * raw_ipd + (1.0 - alpha) * _smoothed_ipd
    return _smoothed_ipd


def nothing(x):
    pass


class CalibrationSession:
    """
    Collects a target number of valid, high-confidence IPD samples across
    varying head and card angles while rejecting single-frame flicker outliers.
    Computes and displays final min, max, mean, std, and angle variety.
    """
    def __init__(self, target_samples=300):
        self.target_samples = target_samples
        self.active = False
        self.complete = False
        self.samples = []
        self.recent_ipds = []
        self.summary = None
        self.rejections = 0

    def start(self):
        self.active = True
        self.complete = False
        self.samples = []
        self.recent_ipds = []
        self.summary = None
        self.rejections = 0
        print("\n" + "=" * 65)
        print(">>> CALIBRATION RECORDING STARTED")
        print(f"    Target: {self.target_samples} samples across varying head & card angles")
        print("    Flicker filter: Active (rejects abrupt outlier frames)")
        print("=" * 65 + "\n")

    def cancel(self):
        if self.active:
            print(">>> Calibration recording cancelled.")
        self.active = False

    def reset(self):
        self.active = False
        self.complete = False
        self.samples = []
        self.recent_ipds = []
        self.summary = None
        self.rejections = 0

    def add_sample(self, cal_ipd, raw_ipd, angle, depth_offset):
        if not self.active or self.complete:
            return False

        if cal_ipd is None or not (45.0 <= cal_ipd <= 85.0):
            self.rejections += 1
            return False

        # Anti-flicker / outlier filter:
        # After collecting a few initial frames, ensure the measurement does not
        # abruptly deviate from the moving median of recent stable frames.
        if len(self.recent_ipds) >= 8:
            med = float(np.median(self.recent_ipds[-20:]))
            if abs(cal_ipd - med) > 1.8:
                self.rejections += 1
                return False

        self.samples.append({
            "ipd": float(cal_ipd),
            "raw_ipd": float(raw_ipd),
            "angle": float(angle),
            "depth": float(depth_offset),
        })
        self.recent_ipds.append(float(cal_ipd))
        if len(self.recent_ipds) > 30:
            self.recent_ipds.pop(0)

        if len(self.samples) >= self.target_samples:
            self._finalize()
            return True

        return True

    def _finalize(self):
        self.active = False
        self.complete = True
        ipd_vals = np.array([s["ipd"] for s in self.samples])
        raw_vals = np.array([s["raw_ipd"] for s in self.samples])
        angles = np.array([s["angle"] for s in self.samples])

        mean_ipd = float(np.mean(ipd_vals))
        min_ipd = float(np.min(ipd_vals))
        max_ipd = float(np.max(ipd_vals))
        std_ipd = float(np.std(ipd_vals))
        angle_span = float(np.ptp(angles))

        self.summary = {
            "count": len(ipd_vals),
            "mean": mean_ipd,
            "min": min_ipd,
            "max": max_ipd,
            "std": std_ipd,
            "range": max_ipd - min_ipd,
            "angle_min": float(np.min(angles)),
            "angle_max": float(np.max(angles)),
            "angle_span": angle_span,
            "rejections": self.rejections,
        }

        print("\n" + "=" * 65)
        print(f">>> CALIBRATION COMPLETE ({len(ipd_vals)} SAMPLES)")
        print(f"    Mean IPD : {mean_ipd:.2f} mm  (std: {std_ipd:.2f} mm)")
        print(f"    Min IPD  : {min_ipd:.2f} mm   | Max IPD : {max_ipd:.2f} mm")
        print(f"    Range    : {max_ipd - min_ipd:.2f} mm")
        print(f"    Angles   : {np.min(angles):+.1f} deg to {np.max(angles):+.1f} deg (span: {angle_span:.1f} deg)")
        print("=" * 65 + "\n")

        try:
            cv2.setTrackbarPos("Calibrate (0/1)", CONTROLS_WINDOW, 0)
        except Exception:
            pass


_calib_session = CalibrationSession(target_samples=300)


def setup_windows():
    cv2.namedWindow(CONTROLS_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROLS_WINDOW, 460, 750)
    cv2.moveWindow(CONTROLS_WINDOW, 30, 30)

    cv2.namedWindow(FEED_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(FEED_WINDOW, 1100, 825)
    cv2.moveWindow(FEED_WINDOW, 510, 30)

    cv2.createTrackbar("Freeze (0/1)", CONTROLS_WINDOW, 0, 1, nothing)
    cv2.createTrackbar("Calibrate (0/1)", CONTROLS_WINDOW, 0, 1, nothing)
    cv2.createTrackbar("Blur", CONTROLS_WINDOW, 7, 15, nothing)
    cv2.createTrackbar("Canny Low", CONTROLS_WINDOW, 35, 255, nothing)
    cv2.createTrackbar("Canny High", CONTROLS_WINDOW, 70, 255, nothing)
    cv2.createTrackbar("Theta (pi/X)", CONTROLS_WINDOW, 549, 720, nothing)
    cv2.createTrackbar("Hough Thresh", CONTROLS_WINDOW, 42, 200, nothing)
    cv2.createTrackbar("Min Length", CONTROLS_WINDOW, 20, 200, nothing)
    cv2.createTrackbar("Max Gap", CONTROLS_WINDOW, 25, 150, nothing)
    cv2.createTrackbar("Angle Tol", CONTROLS_WINDOW, 7, 30, nothing)
    cv2.createTrackbar("Corner Slack", CONTROLS_WINDOW, 150, 250, nothing)
    cv2.createTrackbar("Overlap (%)", CONTROLS_WINDOW, 0, 100, nothing)
    cv2.createTrackbar("Infer 4th (0/1)", CONTROLS_WINDOW, 1, 1, nothing)


def get_controls():
    calib_trackbar = bool(cv2.getTrackbarPos("Calibrate (0/1)", CONTROLS_WINDOW))
    if calib_trackbar and not _calib_session.active and not _calib_session.complete:
        _calib_session.start()
    elif not calib_trackbar and _calib_session.active:
        _calib_session.cancel()

    return {
        "freeze": bool(cv2.getTrackbarPos("Freeze (0/1)", CONTROLS_WINDOW)),
        "calibrate": _calib_session.active,
        "blur": cv2.getTrackbarPos("Blur", CONTROLS_WINDOW),
        "canny_low": cv2.getTrackbarPos("Canny Low", CONTROLS_WINDOW),
        "canny_high": cv2.getTrackbarPos("Canny High", CONTROLS_WINDOW),
        "theta_div": max(1, cv2.getTrackbarPos("Theta (pi/X)", CONTROLS_WINDOW)),
        "hough_thresh": max(1, cv2.getTrackbarPos("Hough Thresh", CONTROLS_WINDOW)),
        "min_length": max(1, cv2.getTrackbarPos("Min Length", CONTROLS_WINDOW)),
        "max_gap": cv2.getTrackbarPos("Max Gap", CONTROLS_WINDOW),
        "angle_tol": max(1, cv2.getTrackbarPos("Angle Tol", CONTROLS_WINDOW)),
        "corner_slack": cv2.getTrackbarPos("Corner Slack", CONTROLS_WINDOW),
        "overlap_pct": max(0, cv2.getTrackbarPos("Overlap (%)", CONTROLS_WINDOW)) / 100.0,
        "infer_4th": bool(cv2.getTrackbarPos("Infer 4th (0/1)", CONTROLS_WINDOW)),
    }


def toggle_freeze():
    curr = cv2.getTrackbarPos("Freeze (0/1)", CONTROLS_WINDOW)
    cv2.setTrackbarPos("Freeze (0/1)", CONTROLS_WINDOW, 0 if curr else 1)


def toggle_calibrate():
    curr = cv2.getTrackbarPos("Calibrate (0/1)", CONTROLS_WINDOW)
    new_val = 0 if curr else 1
    cv2.setTrackbarPos("Calibrate (0/1)", CONTROLS_WINDOW, new_val)
    if new_val:
        _calib_session.start()
    else:
        _calib_session.cancel()


def fit_to_window(img, win_w, win_h):
    if win_w <= 10 or win_h <= 10:
        return img
    h, w = img.shape[:2]
    scale = min(win_w / w, win_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    yo = (win_h - nh) // 2
    xo = (win_w - nw) // 2
    canvas[yo:yo + nh, xo:xo + nw] = resized
    return canvas


def draw_debug_overlay(frame_edges, lines, quads, best_quad, left_pupil=None, right_pupil=None, face_roi=None, depth_offset=0.0, freeze=False):
    debug_frame = cv2.cvtColor(frame_edges, cv2.COLOR_GRAY2BGR)

    # Face / forehead ROI outline: yellow
    if face_roi is not None:
        rx1, ry1, rx2, ry2 = face_roi
        cv2.rectangle(debug_frame, (rx1, ry1), (rx2, ry2), (255, 200, 0), 1)

    # 1. Raw Hough lines: red
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0] if line.ndim == 2 else line
            cv2.line(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

    # 2. Candidate quads: filled blue + blue outline
    if quads:
        overlay = debug_frame.copy()
        for q in quads:
            cv2.fillPoly(overlay, [q.reshape(-1, 1, 2)], (255, 0, 0))
        cv2.addWeighted(overlay, 0.3, debug_frame, 0.7, 0, debug_frame)

        for q in quads:
            cv2.drawContours(debug_frame, [q.reshape(-1, 1, 2)], 0, (255, 0, 0), 2)

    card_long_side = None
    # 3. Winner: green
    if best_quad is not None:
        cv2.drawContours(debug_frame, [best_quad.reshape(-1, 1, 2)], 0, (0, 255, 0), 3)

        # Perspective normalized long side
        card_long_side = CR80Detect.get_perspective_flattened_long_side(best_quad)

        cx = int(np.mean(best_quad[:, 0]))
        cy = int(np.mean(best_quad[:, 1]))
        cv2.putText(
            debug_frame,
            f"Card (flat): {card_long_side:.1f} px",
            (cx - 65, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # 4. Pupils: circles + line + IPD measurement
    if left_pupil is not None:
        cv2.circle(debug_frame, left_pupil, 4, (0, 0, 255), -1)
    if right_pupil is not None:
        cv2.circle(debug_frame, right_pupil, 4, (0, 0, 255), -1)
    if left_pupil is not None and right_pupil is not None:
        cv2.line(debug_frame, left_pupil, right_pupil, (255, 255, 0), 1)
        pupil_dist = math.hypot(right_pupil[0] - left_pupil[0], right_pupil[1] - left_pupil[1])
        mid_x = (left_pupil[0] + right_pupil[0]) // 2
        mid_y = (left_pupil[1] + right_pupil[1]) // 2

        cv2.putText(
            debug_frame,
            f"Pupils: {pupil_dist:.1f} px",
            (mid_x - 50, mid_y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )

        if card_long_side and card_long_side > 1e-3:
            calibrated_ipd = CR80Detect.compute_calibrated_ipd(pupil_dist, card_long_side, depth_offset=depth_offset)
            raw_ipd = (pupil_dist / card_long_side) * CR80Detect.CR80_LONG_SIDE_MM
            ipd_mm = get_smoothed_ipd(calibrated_ipd)
            cv2.putText(
                debug_frame,
                f"IPD: {ipd_mm:.1f} mm",
                (mid_x - 60, mid_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 255),
                2,
            )

            # Record sample if calibration session is active
            if _calib_session.active:
                card_angle = 0.0
                if best_quad is not None:
                    rect = cv2.minAreaRect(best_quad.astype(np.float32))
                    card_angle = rect[2]
                    if rect[1][0] < rect[1][1]:
                        card_angle += 90.0
                    card_angle = ((card_angle + 90) % 180) - 90
                _calib_session.add_sample(calibrated_ipd, raw_ipd, card_angle, depth_offset)

    # 5. Calibration Progress & Summary HUD
    h, w = debug_frame.shape[:2]
    cx = w // 2

    if _calib_session.active:
        card_w, card_h = 660, 115
        x1, y1 = max(10, cx - card_w // 2), 15
        x2, y2 = min(w - 10, x1 + card_w), y1 + card_h

        overlay = debug_frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.82, debug_frame, 0.18, 0, debug_frame)
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 220, 255), 2)

        num_s = len(_calib_session.samples)
        target_s = _calib_session.target_samples
        pct = min(1.0, num_s / float(target_s))

        cv2.putText(debug_frame, "CALIBRATING IPD (VARIETY SAMPLING)", (x1 + 20, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 2)
        cv2.putText(debug_frame, f"{num_s} / {target_s} ({int(pct * 100)}%)", (x2 - 170, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

        # Progress bar
        bx1, by1, bx2, by2 = x1 + 20, y1 + 40, x2 - 20, y1 + 62
        fill_w = int((bx2 - bx1) * pct)
        cv2.rectangle(debug_frame, (bx1, by1), (bx1 + fill_w, by2), (0, 230, 115), -1)
        cv2.rectangle(debug_frame, (bx1, by1), (bx2, by2), (180, 180, 180), 1)

        # Guidance and live stats
        angles_so_far = [s["angle"] for s in _calib_session.samples] if _calib_session.samples else [0.0]
        span_so_far = float(np.ptp(angles_so_far))
        live_str = f"{calibrated_ipd:.1f} mm" if (left_pupil and card_long_side) else "Searching card..."
        curr_angle = 0.0
        if best_quad is not None:
            rect = cv2.minAreaRect(best_quad.astype(np.float32))
            curr_angle = rect[2]
            if rect[1][0] < rect[1][1]:
                curr_angle += 90.0
            curr_angle = ((curr_angle + 90) % 180) - 90
        cv2.putText(debug_frame, f"Live: {live_str} | Roll: {curr_angle:+.1f} deg | Angle Spread: {span_so_far:.1f} deg", (x1 + 20, y1 + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 200), 1)
        cv2.putText(debug_frame, "Slowly tilt head / card to capture diverse angles", (x1 + 20, y1 + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 220, 255), 1)

    elif _calib_session.complete and _calib_session.summary:
        summary = _calib_session.summary
        card_w, card_h = 720, 185
        x1, y1 = max(10, cx - card_w // 2), 15
        x2, y2 = min(w - 10, x1 + card_w), y1 + card_h

        overlay = debug_frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 18, 25), -1)
        cv2.addWeighted(overlay, 0.88, debug_frame, 0.12, 0, debug_frame)
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 120), 2)

        cv2.putText(debug_frame, f"CALIBRATION COMPLETE  [{summary['count']} VALID SAMPLES]", (x1 + 25, y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 120), 2)
        cv2.putText(debug_frame, f"MEAN IPD:  {summary['mean']:.1f} mm", (x1 + 25, y1 + 75), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (0, 255, 255), 3)
        cv2.putText(debug_frame, f"Std Dev: +/- {summary['std']:.2f} mm", (x1 + 420, y1 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 200, 255), 2)
        cv2.putText(debug_frame, f"Min: {summary['min']:.1f} mm   |   Max: {summary['max']:.1f} mm   |   Range: {summary['range']:.1f} mm", (x1 + 25, y1 + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (240, 240, 240), 2)
        cv2.putText(debug_frame, f"Angle Variety: {summary['angle_min']:+.1f} deg to {summary['angle_max']:+.1f} deg (Total Spread: {summary['angle_span']:.1f} deg)", (x1 + 25, y1 + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 220, 150), 1)
        cv2.putText(debug_frame, "Press 'C' or toggle Calibrate trackbar to run a new session", (x1 + 25, y1 + 172), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (170, 170, 170), 1)

    else:
        cv2.putText(debug_frame, "Press 'C' or toggle Calibrate trackbar to start 300-frame IPD calibration", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

    if freeze:
        cv2.putText(debug_frame, "FROZEN (Press 'F' or Space to toggle)", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return debug_frame


def show(debug_frame):
    controls_bg = np.zeros((10, 460, 3), dtype=np.uint8)
    cv2.imshow(CONTROLS_WINDOW, controls_bg)

    rect = cv2.getWindowImageRect(FEED_WINDOW)
    display_frame = fit_to_window(debug_frame, rect[2], rect[3]) if rect[2] > 10 and rect[3] > 10 else debug_frame
    cv2.imshow(FEED_WINDOW, display_frame)
