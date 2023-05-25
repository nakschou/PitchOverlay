import vision_processing as vp
import bounding_box_projection as bbp
import predictor as pred
import pandas as pd
import overlay as ov
from ultralytics import YOLO

model_path = 'runs/detect/pitch_detection_v32/weights/best.pt'
vid1_path = "pitcher_vids/gallen1.mp4"
vid2_path = "pitcher_vids/gallen2.mp4"
pitch1_velo = 89
pitch2_velo = 83
boxes_path = "csvs/gallen1.csv"
boxes2_path = "csvs/gallen2.csv"
poly_deg = 3
out1_path = "processed_vids/cole1tracked.mp4"
out2_path = "processed_vids/cole2tracked.mp4"
final_outpath = "processed_vids/overlay.mp4"

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
    toi = bbp.get_toi(vid_data, pitch1_velo, df)
    #shrinks the dataframe to the time of interest
    df = df[(df['frame'] >= toi[0]) & (df['frame'] <= toi[1])]
    df = bbp.eliminate_outliers(df)
    df = bbp.normalize_boxes(df, toi)
    df.to_csv(boxes_path)
    dct = vp.get_circles(df, vid1_path, out1_path)
    ov.overlay_video(out1_path, vid2_path, toi, final_outpath, boxes_path)
