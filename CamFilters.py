import cv2


def apply_filters(frame, blur_val, canny_low, canny_high):
    k = max(1, blur_val if blur_val % 2 == 1 else blur_val + 1)
    canny_high_val = max(canny_low + 1, canny_high)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray, (k, k), 0)
    frame_edges = cv2.Canny(frame_blur, canny_low, canny_high_val)

    return frame_edges
