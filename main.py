import cv2
import numpy as np

camera = cv2.VideoCapture(0)

while True:
    (ret, frame) = camera.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    frame = cv2.Canny(frame, 50, 150)

    lines = cv2.HoughLinesP(frame, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=30)

    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)   # red now possible

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow("OpenIPD Alpha", frame)
    if cv2.waitKey(1) == 27:
        break

