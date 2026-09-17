import cv2
import numpy

camera = cv2.VideoCapture(0)

while True:
    (ret, frame) = camera.read()

    cv2.imshow("OpenIPD Alpha", frame)
    if cv2.waitKey(1) == 27:
        break

