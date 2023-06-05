from ultralytics import YOLO
import numpy as np
import cv2 as cv
import pandas as pd
import config as cfg
import utility as ut
import os

model_path = cfg.fileConfig.model_path
vid_path = ut.video_path("kershaw1",
                                 cfg.fileConfig.pitcher_vids_path)

model = YOLO(model_path)
model.track(vid_path, save=True, conf=0.03)