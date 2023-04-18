from __future__ import print_function
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from tracker import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='pitcher_vids/lhp_sl.mp4')
parser.add_argument('--algo', type=str, help='Background subtraction \
method (KNN, MOG2).', default='KNN')
args = parser.parse_args()
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()
cap = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))

while cap.isOpened():
    _, image = cap.read()
    if image is None:
        break
    
    # convert to grayscale
    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    grayscale = cv.medianBlur(grayscale, 5)
    fgMask = backSub.apply(grayscale)
    
    # detect circles via hough
    circles = cv.HoughCircles(image=fgMask, method=cv.HOUGH_GRADIENT, dp=1, 
                              minRadius=5, minDist=80, param1=5, param2=10, 
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
    cv.imshow('FG Mask', fgMask)
    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()