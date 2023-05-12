import pandas as pd
import cv2 as cv
import numpy as np

path = "pitcher_vids/pitcher (3).mp4"
boxes_path = 'boxes2.csv'
out_path = "tracker2.mp4"
desired_timeframe = 0.5 #timeframe in seconds
conf_ind = 7
size_increment = 3

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

#helper function for get_toi        
def create_frame_arr(df, length):
    frame_arr = np.zeros(length+1)
    prevnum = df.at[0,'frame']
    maxconf_prevnum = df.at[0, 'confidence']
    for i in range(1, df.shape[0]):
        print(prevnum)
        if df.at[i, 'frame'] != prevnum:
            frame_arr[prevnum] += maxconf_prevnum
            maxconf_prevnum = df.at[i, 'confidence']
        prevnum = df.at[i, 'frame']
        if(df.at[i, 'confidence'] > maxconf_prevnum):
            maxconf_prevnum = df.at[i, 'confidence']
    #print(frame_arr)
    return frame_arr

def eliminate_outliers(df):
    confdf = df[df['confidence'] > 0.5]
    x_parametric = np.polyfit(confdf['frame'], confdf['x_center'], 2)
    y_parametric = np.polyfit(confdf['frame'], confdf['y_center'], 2)
    df['x_dist'] = df.apply(lambda row: dist(row['x_center'], row['frame'], \
                                                  x_parametric), axis=1)
    df['y_dist'] = df.apply(lambda row: dist(row['y_center'], row['frame'], \
                                                    y_parametric), axis=1)
    df = df[abs(df['x_dist']) < 10]
    df = df[abs(df['y_dist']) < 10]
    df = df.drop('x_dist', axis=1)
    df = df.drop('y_dist', axis=1)
    df = df.drop('confidence', axis=1)
    return df

def dist(var, frame, var_parametric):
    return var-(var_parametric[0]*frame**2 + var_parametric[1]*frame + \
              var_parametric[2])

def normalize_boxes(df):
    df['x_size'] = (df['x2'] - df['x1'])
    df['y_size'] = (df['y2'] - df['y1'])
    newxsize = df['x_size'].mean() + size_increment
    newysize = df['y_size'].mean() + size_increment
    df = df.drop('x_size', axis=1)
    df = df.drop('y_size', axis=1)
    x_parametric = np.polyfit(df['frame'], df['x_center'], 2)
    y_parametric = np.polyfit(df['frame'], df['y_center'], 2)
    framemax = df['frame'].max()
    framearr = np.zeros(framemax+1)
    for i in df['frame']:
        framearr[i] = 1
    missing_rows = {}
    for i in range(len(framearr)):
        if(framearr[i] == 0):
            missing_rows[i] = [0, 0, 0, 0, x_parametric[0]*i**2 + \
                            x_parametric[1]*i + x_parametric[2], 
                            y_parametric[0]*i**2 + y_parametric[1]*i + \
                            y_parametric[2]]
    missing_df = pd.DataFrame.from_dict(missing_rows, orient='index', \
                                        columns=['x1', 'y1', 'x2',\
                                'y2', 'x_center', 'y_center'])
    #print(df)
    missing_df.index.name = 'frame'
    missing_df.reset_index(inplace=True)
    df = pd.concat([df, missing_df], ignore_index=True)
    df['x1'] = df['x_center'] - newxsize/2
    df['x2'] = df['x_center'] + newxsize/2
    df['y1'] = df['y_center'] - newysize/2
    df['y2'] = df['y_center'] + newysize/2
    return df


def video_with_boxes(df, vid_path, out_path):
    cap = cv.VideoCapture(vid_path)
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    framerate = int(cap.get(cv.CAP_PROP_FPS))
    fourcc = cv.VideoWriter_fourcc(*'mp4v') # Specify the codec
    out = cv.VideoWriter(out_path, fourcc, framerate, (int(cap.get(3)), int(cap.get(4)))) # Create a video writer object
    for i in range(length):
        ret, frame = cap.read()
        if ret == True:
            if i in df['frame'].values:
                row = df.loc[df['frame'] == i]
                x1 = int(row['x1'])
                y1 = int(row['y1'])
                x2 = int(row['x2'])
                y2 = int(row['y2'])
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            out.write(frame) # Write the modified frame to the output video
            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    out.release() # Release the video writer object
    cv.destroyAllWindows()


if __name__ == "__main__":
    df = pd.read_csv(boxes_path)
    df = add_center(df, 'x1', 'y1', 'x2', 'y2')
    vid_data = read_video_data(path)
    print(df)
    toi = get_toi(vid_data, desired_timeframe, df)
    print(toi)
    df = df[(df['frame'] >= toi[0]) & (df['frame'] <= toi[1])]
    df['frame'] = df['frame'] - toi[0]
    df = eliminate_outliers(df)
    df = normalize_boxes(df)
    df['frame'] = df['frame'] + toi[0]
    #print(df)
    video_with_boxes(df, path, out_path)
    #df.to_csv('boxes.csv', index=False)