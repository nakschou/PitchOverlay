import pandas as pd
import cv2 as cv
import numpy as np
import utility as ut
import config as cfg
import os
import math

path = ut.video_path(cfg.fileConfig.pitch1_name, 
                           cfg.fileConfig.pitcher_vids_path)
boxes_path = ut.csv_path_suffix(cfg.fileConfig.pitch1_name, 
                                cfg.fileConfig.csv_path, 
                                cfg.fileConfig.predictor_suffix)
out_path = ut.video_path_suffix(cfg.fileConfig.pitch1_name, 
                           cfg.fileConfig.processed_vids_path,
                           cfg.fileConfig.boxes_suffix)
new_boxes_path = ut.csv_path_suffix(cfg.fileConfig.pitch1_name,
                                    cfg.fileConfig.csv_path,
                                    cfg.fileConfig.boxes_suffix)
pitch_velo = cfg.fileConfig.pitch1_velo
start_frame = cfg.fileConfig.release1_frame
poly_deg = cfg.fileConfig.poly_deg

def read_video_data(path: str) -> tuple:
    """
    Gets the framerate and length of a video.

    Args:
        path (str): Path to video file.
    
    Returns:
        tuple: (framerate, length)
    
    Raises:
        ValueError: If path is invalid.
    """
    if not os.path.isfile(path):
        raise ValueError("Invalid path")
    cap = cv.VideoCapture(path)
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    #print("length: ", length)
    framerate = int(cap.get(cv.CAP_PROP_FPS))
    cap.release()
    return (framerate, length)

def add_center(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the x_center and y_center columns to a dataframe.

    Args:
        df (DataFrame): Dataframe containing bounding box data.
    
    Returns:
        DataFrame: Dataframe with x_center and y_center columns added.
    """
    df['x_center'] = (df['x1'] + df['x2']) / 2
    df['y_center'] = (df['y1'] + df['y2']) / 2
    return df

#Some unused functionality that could be useful in the future if I want
#to completely avoid user input
def get_toi(vid_data: tuple, velo: int, df: pd.DataFrame, start_frame: int) -> \
    tuple:
    """
    Gets a "timeframe of interest" (TOI) for each pitch.

    Given a desired timeframe length, this function finds the window of time
    where the most confident detections occur. If there are multiple 
    windows with the same number of detections, it takes the middle one.

    Args:
        vid_data (tuple): (framerate, length) of video.
        pitch_velo (float): pitch velo in mph
        df (DataFrame): Dataframe containing bounding box data.
    
    Returns:
        tuple: (start, end) of TOI.
    
    Raises:
        ValueError: If toi is longer than the video.
    """
    window = ut.pitch_time_frames(velo)
    if window > vid_data[1]:
        raise ValueError("Desired timeframe is longer than video")
    if(start_frame < 0):
        frame_arr = create_frame_arr(df, vid_data[1])
        #gets the number of detections in first window 
        curr_detections = 0
        for i in range(window):
            curr_detections += frame_arr[i]
        max_detections = curr_detections
        max_detections_i = 0
        #finds the max number of detections in a window, if there are multiple
        #windows with the same number of detections, it takes the middle one  
        for i in range(1, vid_data[1]-window):
            curr_detections -= frame_arr[i-1]
            curr_detections += frame_arr[i+window-1]
            if curr_detections > max_detections:
                max_detections = curr_detections
                max_detections_i = i
        return (max_detections_i, max_detections_i + window)
    else:
        return (start_frame, start_frame + window)

def create_frame_arr(df: pd.DataFrame, length: int) -> np.ndarray:
    """
    Helper function for get_toi. 
    
    Creates an array of length length+1 where each index represents a frame
    and the value at that index represents the highest confidence of
    a detection in that frame. Used to find the most confident timeframe
    in get_toi.

    Args:
        df (DataFrame): Dataframe containing bounding box data.
        length (int): Length of video in frames.
    
    Returns:
        numpy.ndarray: Array of length length+1.
    """
    frame_arr = np.zeros(length+1)
    prevnum = df.at[0,'frame']
    maxconf_prevnum = df.at[0, 'confidence']
    for i in range(1, df.shape[0]):
        if df.at[i, 'frame'] != prevnum:
            frame_arr[prevnum] += maxconf_prevnum
            maxconf_prevnum = df.at[i, 'confidence']
        prevnum = df.at[i, 'frame']
        #if confidence of current detection is higher than previous max, update
        if(df.at[i, 'confidence'] > maxconf_prevnum):
            maxconf_prevnum = df.at[i, 'confidence']
    return frame_arr

def eliminate_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Eliminates the outliers from the dataframe.

    Essentially just removes any detections that are more than 10 pixels
    away from the parametric curve, which is calculated using all points.
    With a confidence above the threshold.

    Args:
        df (DataFrame): Dataframe containing bounding box data.
        conf (int): Confidence threshold.
    
    Returns:
        DataFrame: Dataframe with outliers removed.
    """
    confidence_threshold = df['confidence'].mean()
    confdf = df[df['confidence'] > confidence_threshold]
    #parametricizes the curve of the ball
    x_parametric = np.polyfit(confdf['frame'], confdf['x_center'], poly_deg)
    y_parametric = np.polyfit(confdf['frame'], confdf['y_center'], poly_deg)
    df['x_dist'] = df.apply(lambda row: dist(row['x_center'], row['frame'], \
                                                  x_parametric), axis=1)
    df['y_dist'] = df.apply(lambda row: dist(row['y_center'], row['frame'], \
                                                    y_parametric), axis=1)
    df = df[abs(df['x_dist']) < 20]
    df = df[abs(df['y_dist']) < 20]
    #removes unnecessary columns
    df = df.drop('x_dist', axis=1)
    df = df.drop('y_dist', axis=1)
    df = df.drop('confidence', axis=1)
    df = df.drop('box_num', axis = 1)
    return df

def dist(var: float, frame: int, var_parametric: np.ndarray) -> float:
    """
    Helper function for eliminate_outliers.

    Calculates the distance between a point's x/y value and its parametric
    curve's x/y value.

    Args:
        var (float): x or y value of point.
        frame (int): Frame number of point.
        var_parametric (numpy.ndarray): Parametric curve of x or y values.
    
    Returns:
        float: Distance between point and parametric curve.
    """
    return var-np.polyval(var_parametric, frame)

def normalize_boxes(df: pd.DataFrame, toi: tuple) -> pd.DataFrame:
    """
    Normalizes the bounding boxes in the dataframe.

    Ensures that the size of all boxes is the same (as the model is not very
    good at this), and adds boxes where there are missing detections along
    a parametric curve created by the other points. Parametric curve is 
    recalculated using points with confidence below the threshold included.

    Args:
        df (DataFrame): Dataframe containing bounding box data.
        toi (tuple): (start, end) of timeframe of interest.
    
    Returns:
        DataFrame: Dataframe with normalized boxes.
    """
    x_increment = 2 #make bounding box bigger by x_increment pixels
    y_increment = 0 #make bounding box bigger by y_increment pixels
    #gets the sizes of the boxes
    df['x_size'] = (df['x2'] - df['x1'])
    df['y_size'] = (df['y2'] - df['y1'])
    #sets the new size to be the average size of the boxes + a size increment
    newxsize = df['x_size'].mean() + x_increment
    newysize = df['y_size'].mean() + y_increment
    df = df.drop('x_size', axis=1)
    df = df.drop('y_size', axis=1)
    #new parametric equations
    x_parametric = np.polyfit(df['frame'], df['x_center'], poly_deg)
    y_parametric = np.polyfit(df['frame'], df['y_center'], poly_deg)
    newdf = {}
    for i in range(toi[1] - toi[0]):
        newdf[i+toi[0]] = [0, 0, 0, 0, 
                                np.polyval(x_parametric, i+toi[0]),
                                np.polyval(y_parametric, i+toi[0])]
    #print(missing_rows)
    newdf = pd.DataFrame.from_dict(newdf, orient='index', \
                                        columns=['x1', 'y1', 'x2',\
                                'y2', 'x_center', 'y_center'])
    newdf.index.name = 'frame'
    newdf.reset_index(inplace=True)
    #sets new box xyxy values
    newdf['x1'] = newdf['x_center'] - newxsize/2
    newdf['x2'] = newdf['x_center'] + newxsize/2
    newdf['y1'] = newdf['y_center'] - newysize/2
    newdf['y2'] = newdf['y_center'] + newysize/2
    newdf.sort_values(by=['frame'], inplace=True)
    newdf.reset_index(inplace=True)
    return newdf

def video_with_boxes(df: pd.DataFrame, vid_path: str, out_path: str):
    """
    Creates a video with bounding boxes around the ball.

    Args:
        df (DataFrame): Dataframe containing bounding box data.
        vid_path (str): Path to video.
        out_path (str): Path to output video.
    
    Returns:
        None
    
    Raises:
        ValueError: If vid_path is not a valid path.
    """
    if not os.path.isfile(vid_path):
        raise ValueError("Invalid path")
    cap = cv.VideoCapture(vid_path)
    framerate = int(cap.get(cv.CAP_PROP_FPS))
    #specifies the codec and creates a video writer object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(out_path, fourcc, framerate, (int(cap.get(3)), \
                                                       int(cap.get(4))))
    curr_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            curr_frame = cap.get(cv.CAP_PROP_POS_FRAMES)
            if curr_frame in df['frame'].values:
                row = df.loc[df['frame'] == curr_frame]
                x1 = int(row['x1'].iloc[0])
                y1 = int(row['y1'].iloc[0])
                x2 = int(row['x2'].iloc[0])
                y2 = int(row['y2'].iloc[0])
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Write the modified frame to the output video
            out.write(frame) 
            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    # Release the video writer object
    out.release() 
    cv.destroyAllWindows()

def add_pixel(df: pd.DataFrame, pixel: tuple, frame: int) -> pd.DataFrame:
    """
    Adds the pixel to the dataframe.

    Args:
        df (DataFrame): Dataframe containing bounding box data.
        pixel (tuple): Tuple containing the pixel values.
    
    Returns:
        DataFrame: Dataframe with pixel values added.
    """
    if frame not in df['frame'].values:
        x1 = pixel[0] - cfg.fileConfig.box_size/2
        y1 = pixel[1] - cfg.fileConfig.box_size/2
        x2 = pixel[0] + cfg.fileConfig.box_size/2
        y2 = pixel[1] + cfg.fileConfig.box_size/2
        x_center = pixel[0]
        y_center = pixel[1]
        new_row = {'index': len(df),
                   'frame': frame,
                   'x1': x1,
                   'y1': y1,
                   'x2': x2,
                   'y2': y2,
                   'x_center': x_center,
                   'y_center': y_center}
        df = df.append(new_row, ignore_index=True)
        df = df.sort_values(by=['frame'])
        df = df.reset_index()
        return df
    #add a row to the dataframe with the pixel values
    target_row = df[df['frame'] == frame].index
    df.loc[target_row, 'x_center'] = pixel[0]
    df.loc[target_row, 'y_center'] = pixel[1]
    return df

def this_runner(boxes_path: str, vid_path: str, pitch_velo: int, 
               new_boxes_path: str) -> None:
    """
    Runner for the bounding box projection.

    Adds the center to the csv, finds the timeframe of interest, eliminates
    outliers, normalizes the boxes, and creates a video with the boxes. It then
    writes the new boxes to a csv.

    Args:
        boxes_path (str): Path to csv containing bounding box data.
        vid_path (str): Path to video.
        pitch_velo (int): Pitch velocity in mph.
        new_boxes_path (str): Path to save new boxes to.
    
    Returns:
        None
    """
    df = pd.read_csv(boxes_path)
    df = add_center(df)
    vid_data = read_video_data(vid_path)
    if(cfg.fileConfig.release1_frame < 0):
        tup = ut.get_release_frame(vid_path)
        start_frame = tup[0]
        pixel = tup[1]
    else:
        start_frame = cfg.fileConfig.release1_frame 
    #print(start_frame)
    toi = get_toi(vid_data, pitch_velo, df, start_frame)
    df = df[(df['frame'] >= toi[0]) & (df['frame'] <= toi[1])]
    df = eliminate_outliers(df)
    #print(df)
    df = normalize_boxes(df, toi)
    video_with_boxes(df, path, out_path)
    df.to_csv(new_boxes_path, index=False)
    
if __name__ == "__main__":
    this_runner(boxes_path, path, pitch_velo, new_boxes_path)
    