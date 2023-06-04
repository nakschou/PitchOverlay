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
start1_frame = cfg.fileConfig.release1_frame
start2_frame = cfg.fileConfig.release2_frame
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
    if(cfg.fileConfig.release1_frame < 0):
        tup = ut.get_release_frame(vid1_path)
        start1_frame = tup[0]
        pixel = tup[1]
    else:
        start1_frame = cfg.fileConfig.release1_frame 
    if(cfg.fileConfig.release2_frame < 0):
        tup = ut.get_release_frame(vid2_path)
        start2_frame = tup[0]
        pixel = tup[1]
    else:
        start2_frame = cfg.fileConfig.release2_frame 
        # Sets the timeframe of interest
    toi1 = (start1_frame, start1_frame + ut.pitch_time_frames(pitch1_velo))
    toi2 = (start2_frame, start2_frame + ut.pitch_time_frames(pitch2_velo)) 
    # Gets the boxes in a format unfit for a dataframe
    boxes_dct = pred.get_boxes(model, vid1_path, toi1)
    boxes_dct2 = pred.get_boxes(model, vid2_path, toi2)
    # Converts the boxes to fittable format and writes to dataframe
    df = pred.convert_boxes_df(boxes_dct)
    df2 = pred.convert_boxes_df(boxes_dct2)
    # Adds the center of the boxes to the dataframe
    df = bbp.add_center(df)
    df2 = bbp.add_center(df2)
    # Reads video data
    vid_data = bbp.read_video_data(vid1_path)
    vid2_data = bbp.read_video_data(vid2_path)
    #shrinks the dataframe to the time of interest
    df = df[(df['frame'] >= toi1[0]) & (df['frame'] <= toi1[1])]
    df = bbp.eliminate_outliers(df)
    df = bbp.normalize_boxes(df, toi1)
    df.to_csv(boxes_path)
    df2 = df2[(df2['frame'] >= toi2[0]) & (df2['frame'] <= toi2[1])]
    df2 = bbp.eliminate_outliers(df2)
    df2 = bbp.normalize_boxes(df2, toi2)
    df2.to_csv(boxes2_path)
    vp.get_circles(df, vid1_path, out1_path)
    ov.overlay_video(out1_path, vid2_path, toi1, toi2, final_outpath, 
                     boxes_path, boxes2_path)
