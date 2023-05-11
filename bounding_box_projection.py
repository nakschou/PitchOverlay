import pandas as pd
import cv2 as cv
import numpy as np

path = "pitcher_vids/pitcher (1).mp4"
boxes_path = 'boxes.csv'
desired_timeframe = 0.5 #timeframe in seconds
conf_ind = 7

# Returns (framerate, frame#) of a video
def read_video_data(path):
    cap = cv.VideoCapture(path)
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    framerate = int(cap.get(cv.CAP_PROP_FPS))
    cap.release()
    return (framerate, length)

# Adds a center coordinate for each bounding box
def add_center(df, x1, y1, x2, y2):
    df['x_center'] = (df[x1] + df[x2]) / 2
    df['y_center'] = (df[y1] + df[y2]) / 2
    return df

# Gets a "timeframe of interest" (TOI) for each pitch
# Essentially, finds x (s) window where pitch is being thrown
def get_toi(vid_data, desired_timeframe, df):
    window = int(vid_data[0] * desired_timeframe)
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
        curr_detections += frame_arr[i+window]
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

#helper function for get_toi        
def create_frame_arr(df, length):
    frame_arr = np.zeros(length)
    prevnum = df.at[0,'frame']
    maxconf_prevnum = df.at[0, 'confidence']
    for i in range(1, df.shape[0]):
        #print(type(i))
        if df.at[i, 'frame'] != prevnum:
            frame_arr[prevnum] += maxconf_prevnum
            maxconf_prevnum = df.at[i, 'confidence']
        prevnum = df.at[i, 'frame']
        if(df.at[i, 'confidence'] > maxconf_prevnum):
            maxconf_prevnum = df.at[i, 'confidence']
    print(frame_arr)
    return frame_arr

if __name__ == "__main__":
    df = pd.read_csv('boxes.csv')
    df = add_center(df, 'x1', 'y1', 'x2', 'y2')
    vid_data = read_video_data(path)
    toi = get_toi(vid_data, desired_timeframe, df)
    #print(df)
    print(toi)
    #df.to_csv('boxes.csv', index=False)