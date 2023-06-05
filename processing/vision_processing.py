import pandas as pd
import cv2 as cv
import numpy as np
import config as cfg
import utility as ut
import os

path = ut.video_path(cfg.fileConfig.pitch1_name,
                           cfg.fileConfig.pitcher_vids_path)
boxes_path = ut.csv_path_suffix(cfg.fileConfig.pitch1_name,
                                 cfg.fileConfig.csv_path,
                                 cfg.fileConfig.boxes_suffix)
out_path = ut.video_path_suffix(cfg.fileConfig.pitch1_name,
                                 cfg.fileConfig.processed_vids_path,
                                 cfg.fileConfig.masks_suffix)
poly_deg = cfg.fileConfig.poly_deg

def get_circles(df: pd.DataFrame, vid_path: str, out_path: str) -> None:
    """
    Creates a video with circles around the detected balls.

    Args:
        df (DataFrame): Dataframe containing bounding box data.
        vid_path (str): Path to video.
    
    Returns:
        None
    
    Raises:
        ValueError: If vid_path is not a valid path.
    """
    if not os.path.isfile(path):
        raise ValueError("Invalid path")
    cap = cv.VideoCapture(vid_path)
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    framerate = int(cap.get(cv.CAP_PROP_FPS))
    #specifies the codec and creates a video writer object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(out_path, fourcc, framerate, (int(cap.get(3)), \
                                                       int(cap.get(4))))
    for i in range(length):
        ret, frame = cap.read()
        if ret == True:
            # Define the region of interest (ROI) as a rectangular mask
                mask1 = np.zeros(frame.shape[:2], dtype=np.uint8)
                if i in df['frame'].values:
                    row = df.loc[df['frame'] == i]
                    x1 = int(row['x1'].iloc[0])
                    y1 = int(row['y1'].iloc[0])
                    x2 = int(row['x2'].iloc[0])
                    y2 = int(row['y2'].iloc[0])
                    x_center = int(row['x_center'].iloc[0])
                    y_center = int(row['y_center'].iloc[0])
                    mask1[y1:y2, x1:x2] = 255

                    # create a converted version of the frame
                    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

                    # Get the pixel values at (x, y)
                    pixel = hsv[y_center, x_center]
        
                    lower = pixel - cfg.fileConfig.pixel_low_thres
                    upper = pixel + cfg.fileConfig.pixel_high_thres
                
                    # preparing the mask to overlay
                    mask2 = cv.inRange(hsv, lower, upper)

                    combined_mask = cv.bitwise_and(mask1, mask2)
                    
                    result = cv.bitwise_and(frame, frame, mask = combined_mask)
                    out.write(result)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break
    cap.release()
    # Release the video writer object
    cv.destroyAllWindows()

def vp_runner(boxes_path: str, vid_path: str, out_path: str) -> None:
    """
    Runs the vision processing pipeline.

    Args:
        boxes_path (str): Path to boxes csv.
        vid_path (str): Path to video.
        out_path (str): Path to save video.
    
    Returns:
        None
    """
    df = pd.read_csv(boxes_path)
    get_circles(df, vid_path, out_path)

if __name__ == "__main__":
    vp_runner(boxes_path, path, out_path)