import cv2
import numpy

camera = cv2.VideoCapture(0)

while True:
    (ret, frame) = camera.read()

    frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame  = cv2.GaussianBlur(frame, (5,5), 0)
    frame = cv2.Canny(frame, 50, 150)

    cv2.imshow("OpenIPD Alpha", frame)
    if cv2.waitKey(1) == 27:
        break

