from ultralytics import YOLO
import numpy as np
import cv2 as cv
import pandas as pd
import config as cfg
import utility as ut
import os

model_path = cfg.fileConfig.model_path
vid_path = ut.video_path(cfg.fileConfig.pitch1_name,
                                 cfg.fileConfig.pitcher_vids_path)

model = YOLO(model_path)
# Tracks the video and saves it TODO: eliminate need for this
model.track(vid_path, save=True, conf=0.03)