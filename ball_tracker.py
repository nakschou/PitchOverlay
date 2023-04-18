import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from tracker import *

cap = cv.VideoCapture('pitcher_vids/lhp_sl.mp4')

while cap.isOpened():
    _, image = cap.read()
    # convert to grayscale
    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    grayscale = cv.medianBlur(grayscale, 9)
    edges = cv.Canny(grayscale, 150, 200)
    # detect lines in the image using hough lines technique
    circles = cv.HoughCircles(image=edges, method=cv.HOUGH_GRADIENT, dp=0.3, 
                              minRadius=5, minDist=10, param1=5, param2=10, 
                              maxRadius=10)
    # iterate over the output lines and draw them
    if circles is not None:
        for co, i in enumerate(circles[0, :], start=1):
            # draw the outer circle in green
            cv.circle(image,(int(i[0]),int(i[1])),int(i[2]),(0,255,0),2)
            # draw the center of the circle in red
            cv.circle(image,(int(i[0]),int(i[1])),2,(0,0,255),3)
    cv.imshow("image", image)
    cv.imshow("grayscale", grayscale)
    cv.imshow("edges", edges)
    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()