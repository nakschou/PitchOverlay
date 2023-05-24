import vision_processing as vp
import bounding_box_projection as bbp
import predictor as pred
import pandas as pd
import overlay as ov
import config as cfg
import utility as ut
from ultralytics import YOLO

model_path = cfg.fileConfig.model_path
vid1_path = ut.video_path(cfg.fileConfig.pitch1_name, 
                                cfg.fileConfig.pitcher_vids_path)
vid2_path = ut.video_path(cfg.fileConfig.pitch2_name, 
                                cfg.fileConfig.pitcher_vids_path)
pitch1_velo = cfg.fileConfig.pitch1_velo
pitch2_velo = cfg.fileConfig.pitch2_velo
start1_frame = cfg.fileConfig.start1_frame
start2_frame = cfg.fileConfig.start2_frame
boxes_path = ut.csv_path(cfg.fileConfig.pitch1_name, 
                               cfg.fileConfig.csv_path)
boxes2_path = ut.csv_path(cfg.fileConfig.pitch2_name, 
                                cfg.fileConfig.csv_path)
poly_deg = cfg.fileConfig.poly_deg
out1_path = ut.video_path_suffix(cfg.fileConfig.pitch1_name, 
                                cfg.fileConfig.processed_vids_path,
                                cfg.fileConfig.masks_suffix)
out2_path = ut.video_path_suffix(cfg.fileConfig.pitch2_name, 
                                cfg.fileConfig.processed_vids_path,
                                cfg.fileConfig.masks_suffix)
final_outpath = ut.video_path(cfg.fileConfig.final_file_name, 
                                    cfg.fileConfig.processed_vids_path)

if __name__ == "__main__":
    # Loads the model, change pathing based on what you need
    model = YOLO(model_path)
    # Gets the boxes in a format unfit for a dataframe
    boxes_dct = pred.get_boxes(model, vid1_path)
    # Converts the boxes to fittable format and writes to dataframe
    df = pred.convert_boxes_df(boxes_dct)
    # Adds the center of the boxes to the dataframe
    df = bbp.add_center(df)
    # Reads video data
    vid_data = bbp.read_video_data(vid1_path)
    # Sets the timeframe of interest
    toi = bbp.get_toi(vid_data, pitch1_velo, df, start1_frame)
    #shrinks the dataframe to the time of interest
    df = df[(df['frame'] >= toi[0]) & (df['frame'] <= toi[1])]
    df = bbp.eliminate_outliers(df)
    df = bbp.normalize_boxes(df, toi)
    df.to_csv(boxes_path)
    dct = vp.get_circles(df, vid1_path, out1_path)
    ov.overlay_video(out1_path, vid2_path, toi, final_outpath, boxes_path)
