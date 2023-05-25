import cv2 as cv
import numpy as np
import pandas as pd
import os

def overlay_video(inp1_path: str, inp2_path: str, toi: str, out_path: str, 
                  csv1: str, csv2: str) -> dict:
    """
    Given two videos and a timeframe of interest, overlays the second video
    onto the first video.

    Args:
        df (DataFrame): Dataframe containing bounding box data.
        vid_path (str): Path to video.
    
    Returns:
        dict: Dictionary containing circle data.
    
    Raises:
        ValueError: If vid_path is not a valid path.
    """
    df = pd.read_csv(csv1)
    if not os.path.isfile(inp1_path) or not os.path.isfile(inp2_path):
        raise ValueError("Invalid path")
    cap1 = cv.VideoCapture(inp1_path)
    cap2 = cv.VideoCapture(inp2_path)
    lengthoverlay = int(cap1.get(cv.CAP_PROP_FRAME_COUNT))
    length = int(cap2.get(cv.CAP_PROP_FRAME_COUNT))
    framerate = int(cap2.get(cv.CAP_PROP_FPS))
    #specifies the codec and creates a video writer object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(out_path, fourcc, framerate, (int(cap2.get(3)), \
                                                       int(cap2.get(4))))
    start = False
    firstlines = {}
    firstmin = df['frame'].min()
    for i in range(length):
        if(i == toi[0]):
            start = True
        if(i == toi[0]+lengthoverlay):
            start = False
        ret2, frame2 = cap2.read()
        if ret2 == True:
            if(start):
                if(i-1 >= firstmin):
                    x1 = int(df.loc[df['frame'] == i-1]['x_center'].iloc[0])
                    y1 = int(df.loc[df['frame'] == i-1]['y_center'].iloc[0])
                    x2 = int(df.loc[df['frame'] == i]['x_center'].iloc[0])
                    y2 = int(df.loc[df['frame'] == i]['y_center'].iloc[0])
                    firstlines[i] = ((x1, y1), (x2, y2))
                ret1, frame1 = cap1.read()
                result = cv.bitwise_or(frame1, frame2)
            else:
                result = frame2
            for line in firstlines:
                cv.line(result, firstlines[line][0], firstlines[line][1],
                        (0, 0, 255), 2)
            out.write(result)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap1.release()
    cap2.release()
    # Release the video writer object
    cv.destroyAllWindows()
