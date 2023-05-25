from ultralytics import YOLO
import numpy as np
import cv2 as cv
import pandas as pd
import os

model_path = 'runs/detect/pitch_detection_v42/weights/best.pt'
vid_path = 'pitcher_vids/gallen2.mp4'

model = YOLO(model_path)
# Tracks the video and saves it TODO: eliminate need for this
model.track(vid_path, save=True, conf=0.03)