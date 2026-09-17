import cv2
import numpy

camera = cv2.VideoCapture(0)

while True:
    (ret, frame) = camera.read()

    frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame  = cv2.GaussianBlur(frame, (5,5), 0)
    frame = cv2.Canny(frame, 50, 150)

    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card  = max(contours, key=cv2.contourArea)
    rect  = cv2.minAreaRect(card)
    box   = cv2.boxPoints(rect).astype(int)   # 4 corners of the fit

    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)   # red now possible

    cv2.drawContours(frame, [box], 0, (0, 0, 255), 3)

    cv2.imshow("OpenIPD Alpha", frame)
    if cv2.waitKey(1) == 27:
        break

