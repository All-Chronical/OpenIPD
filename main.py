import cv2
import numpy

camera = cv2.VideoCapture(0)

while True:
    (ret, frame) = camera.read()

    frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame  = cv2.GaussianBlur(frame, (5,5), 0)

    corners = cv2.goodFeaturesToTrack(frame, 50, 0.5, 10)
    if corners is not None:
        for x, y in corners.reshape(-1, 2).astype(int):
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

    cv2.imshow("OpenIPD Alpha", frame)
    if cv2.waitKey(1) == 27:
        break

