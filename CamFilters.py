import cv2
import numpy as np


def apply_filters(frame, blur_val, canny_low, canny_high):
    k = max(1, blur_val if blur_val % 2 == 1 else blur_val + 1)
    canny_high_val = max(canny_low + 1, canny_high)

    # Multi-channel color edge detection: evaluates Canny across individual color channels
    # and merges them via maximum response. This ensures cards of ANY color
    # (white, dark red/maroon, blue, black, metallic, etc.) produce crisp outer boundaries
    # without luminance loss or skin-tone color blending.
    chans = cv2.split(frame)
    edges = [
        cv2.Canny(cv2.GaussianBlur(c, (k, k), 0), canny_low, canny_high_val)
        for c in chans
    ]
    return np.maximum.reduce(edges)


def apply_roi_mask(frame_edges, roi):
    if roi is None:
        return frame_edges
    x1, y1, x2, y2 = roi
    h, w = frame_edges.shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    masked = np.zeros_like(frame_edges)
    masked[y1:y2, x1:x2] = frame_edges[y1:y2, x1:x2]
    return masked
