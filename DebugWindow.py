import math
import cv2
import numpy as np

FEED_WINDOW = "OpenIPD Alpha"
CONTROLS_WINDOW = "Controls"


def nothing(x):
    pass


def setup_windows():
    cv2.namedWindow(CONTROLS_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROLS_WINDOW, 460, 720)
    cv2.moveWindow(CONTROLS_WINDOW, 30, 30)

    cv2.namedWindow(FEED_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(FEED_WINDOW, 1100, 825)
    cv2.moveWindow(FEED_WINDOW, 510, 30)

    cv2.createTrackbar("Freeze (0/1)", CONTROLS_WINDOW, 0, 1, nothing)
    cv2.createTrackbar("Blur", CONTROLS_WINDOW, 8, 15, nothing)
    cv2.createTrackbar("Canny Low", CONTROLS_WINDOW, 55, 255, nothing)
    cv2.createTrackbar("Canny High", CONTROLS_WINDOW, 82, 255, nothing)
    cv2.createTrackbar("Theta (pi/X)", CONTROLS_WINDOW, 549, 720, nothing)
    cv2.createTrackbar("Hough Thresh", CONTROLS_WINDOW, 56, 200, nothing)
    cv2.createTrackbar("Min Length", CONTROLS_WINDOW, 30, 200, nothing)
    cv2.createTrackbar("Max Gap", CONTROLS_WINDOW, 29, 150, nothing)
    cv2.createTrackbar("Angle Tol", CONTROLS_WINDOW, 7, 30, nothing)
    cv2.createTrackbar("Corner Slack", CONTROLS_WINDOW, 57, 150, nothing)
    cv2.createTrackbar("Overlap (%)", CONTROLS_WINDOW, 0, 100, nothing)
    cv2.createTrackbar("Infer 4th (0/1)", CONTROLS_WINDOW, 1, 1, nothing)


def get_controls():
    return {
        "freeze": bool(cv2.getTrackbarPos("Freeze (0/1)", CONTROLS_WINDOW)),
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


def draw_debug_overlay(frame_edges, lines, quads, best_quad, left_pupil=None, right_pupil=None, freeze=False):
    debug_frame = cv2.cvtColor(frame_edges, cv2.COLOR_GRAY2BGR)

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

    # 3. Winner: green
    if best_quad is not None:
        cv2.drawContours(debug_frame, [best_quad.reshape(-1, 1, 2)], 0, (0, 255, 0), 3)

        edges = [best_quad[(i + 1) % 4] - best_quad[i] for i in range(4)]
        lengths = [math.hypot(e[0], e[1]) for e in edges]
        side_a = (lengths[0] + lengths[2]) / 2.0
        side_b = (lengths[1] + lengths[3]) / 2.0
        card_long_side = max(side_a, side_b)

        cx = int(np.mean(best_quad[:, 0]))
        cy = int(np.mean(best_quad[:, 1]))
        cv2.putText(
            debug_frame,
            f"Card: {card_long_side:.1f} px",
            (cx - 50, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # 4. Pupils: circles + line
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
            (mid_x - 50, mid_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

    if freeze:
        cv2.putText(debug_frame, "FROZEN (Press 'F' or Space to toggle)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return debug_frame


def show(debug_frame):
    controls_bg = np.zeros((10, 460, 3), dtype=np.uint8)
    cv2.imshow(CONTROLS_WINDOW, controls_bg)

    rect = cv2.getWindowImageRect(FEED_WINDOW)
    display_frame = fit_to_window(debug_frame, rect[2], rect[3]) if rect[2] > 10 and rect[3] > 10 else debug_frame
    cv2.imshow(FEED_WINDOW, display_frame)
