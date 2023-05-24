from ultralytics import YOLO
import numpy as np
import cv2 as cv
import pandas as pd
import os

model_path = 'ckpt_best.pth'
vid_path = 'pitcher_vids/gallen1.mp4'

best_model = models.get('yolo_nas_l',
                        num_classes=len(dataset_params['classes']),
                        checkpoint_path="yolonas/baseball_yolonas/ckpt_best.pth")
# Tracks the video and saves it TODO: eliminate need for this
model.track(vid_path, save=True, conf=0.03)