import pandas as pd
import cv2 as cv
import numpy as np
import os
import math

path = "pitcher_vids/pitcher (3).mp4"
boxes_path = 'boxes2.csv'
out_path = "tracker2.mp4"
new_boxes_path = 'new_boxes2.csv'
pitch_velo = 82 #mph
poly_deg = 2 #degree of polynomial used for parametric curve

def pitch_time_frames(speed: int) -> int:
    # Convert mph to m/s
    v0 = speed * 0.44704
    # Adjust for release angle
    v0 = v0 * math.cos(math.radians(math.pi/36))

    # Constants
    c_d = 0.47      # Drag coefficient of a sphere
    a = 0.004145    # Cross-sectional area of a baseball (m^2)
    m = 0.145       # Mass of a baseball (kg)
    rho = 1.225     # Density of air at room temperature (kg/m^3)
    dist = 18.44 + 1.28 # Distance from pitcher to plate (m)
    # the weirdo number is to account for the fact that the catcher generally
    # catches the ball a little bit behind the plate

    # Calculate the drag force
    f_d = 0.5 * c_d * a * rho * v0**2

    # Calculate the acceleration due to air resistance
    a_D = f_d / m

    # Calculate the time to reach the plate
    t = (-v0 + math.sqrt(v0**2 + 2*a_D*dist)) / a_D

    return int(t*60+0.5)

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

def get_toi(vid_data: tuple, velo: int, df: pd.DataFrame) -> \
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
    window = pitch_time_frames(velo)
    if window > vid_data[1]:
        raise ValueError("Desired timeframe is longer than video")
    frame_arr = create_frame_arr(df, vid_data[1])
    #gets the number of detections in first window 
    curr_detections = 0
    for i in range(window):
        curr_detections += frame_arr[i]
    max_detections = curr_detections
    max_detections_start = 0
    max_equal_detections = 0
    #finds the maximum number of detections in a window, if there are multiple
    #windows with the same number of detections, it takes the middle one    
    for i in range(1, vid_data[1]-window):
        curr_detections -= frame_arr[i-1]
        curr_detections += frame_arr[i+window-1]
        if curr_detections > max_detections:
            max_detections = curr_detections
            max_detections_start = i
            max_equal_detections = 0
        elif curr_detections == max_detections:
            max_equal_detections += 1
    #print(max_detections_start)
    if max_equal_detections > 0:
        start = (max_detections_start*2 + max_equal_detections) // 2
        return (start, start + window)
    else:
        return (max_detections_start, max_detections_start + window)
     
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
    
    Returns:
        DataFrame: Dataframe with outliers removed.
    """
    confidence_threshold = 0.5 #threshold for points used for parametric curve
    confdf = df[df['confidence'] > confidence_threshold]
    #parametricizes the curve of the ball
    x_parametric = np.polyfit(confdf['frame'], confdf['x_center'], poly_deg)
    y_parametric = np.polyfit(confdf['frame'], confdf['y_center'], poly_deg)
    df['x_dist'] = df.apply(lambda row: dist(row['x_center'], row['frame'], \
                                                  x_parametric), axis=1)
    df['y_dist'] = df.apply(lambda row: dist(row['y_center'], row['frame'], \
                                                    y_parametric), axis=1)
    df = df[abs(df['x_dist']) < 10]
    df = df[abs(df['y_dist']) < 10]
    #removes unnecessary columns
    df = df.drop('x_dist', axis=1)
    df = df.drop('y_dist', axis=1)
    df = df.drop('confidence', axis=1)
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

def normalize_boxes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the bounding boxes in the dataframe.

    Ensures that the size of all boxes is the same (as the model is not very
    good at this), and adds boxes where there are missing detections along
    a parametric curve created by the other points. Parametric curve is 
    recalculated using points with confidence below the threshold included.

    Args:
        df (DataFrame): Dataframe containing bounding box data.
    
    Returns:
        DataFrame: Dataframe with normalized boxes.
    """
    x_increment = 3 #make bounding box bigger by x_increment pixels
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
    x_parametric = np.polyfit(df['frame'], df['x_center'], 2)
    y_parametric = np.polyfit(df['frame'], df['y_center'], 2)
    #adds the new boxes
    framemax = df['frame'].max()
    framearr = np.zeros(framemax+1)
    for i in df['frame']:
        framearr[i] = 1
    missing_rows = {}
    for i in range(len(framearr)):
        if(framearr[i] == 0):
            missing_rows[i] = [0, 0, 0, 0, np.polyval(x_parametric, i),
                               np.polyval(y_parametric, i)]
    missing_df = pd.DataFrame.from_dict(missing_rows, orient='index', \
                                        columns=['x1', 'y1', 'x2',\
                                'y2', 'x_center', 'y_center'])
    missing_df.index.name = 'frame'
    missing_df.reset_index(inplace=True)
    df = pd.concat([df, missing_df], ignore_index=True)
    #sets new box xyxy values
    df['x1'] = df['x_center'] - newxsize/2
    df['x2'] = df['x_center'] + newxsize/2
    df['y1'] = df['y_center'] - newysize/2
    df['y2'] = df['y_center'] + newysize/2
    return df


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
            if i in df['frame'].values:
                row = df.loc[df['frame'] == i]
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

def bbp_runner(boxes_path: str, vid_path: str, pitch_velo: int, 
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
    vid_data = read_video_data(path)
    toi = get_toi(vid_data, pitch_velo, df)
    df = df[(df['frame'] >= toi[0]) & (df['frame'] <= toi[1])]
    df['frame'] = df['frame'] - toi[0]
    df = eliminate_outliers(df)
    df = normalize_boxes(df)
    df['frame'] = df['frame'] + toi[0]
    video_with_boxes(df, path, out_path)
    df.to_csv(new_boxes_path, index=False)
    
if __name__ == "__main__":
    bbp_runner(boxes_path, path, pitch_velo, new_boxes_path)
    