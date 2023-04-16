import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from tracker import *

cap = cv.VideoCapture('pitcher_vids/lhp_cb.mp4')

while cap.isOpened():
    _, image = cap.read()
    # convert to grayscale
    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # perform edge detection
    edges = cv.Canny(grayscale, 30, 100)
    # detect lines in the image using hough lines technique
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 60, np.array([]), 50, 5)
    # iterate over the output lines and draw them
    cv.imshow("image", image)
    cv.imshow("edges", edges)
    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()