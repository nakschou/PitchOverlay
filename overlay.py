import cv2 as cv
import numpy as np
import pandas as pd
import os
import config as cfg

def overlay_video(inp1_path: str, inp2_path: str, toi: tuple, toi2: tuple, 
                  out_path: str, csv1: str, csv2: str) -> dict:
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
    if not os.path.isfile(inp1_path) or not os.path.isfile(inp2_path):
        raise ValueError("Invalid path")
    timelength = int(cfg.fileConfig.line_hold_time*60)
    tracers = cfg.fileConfig.tracers
    thick = cfg.fileConfig.tracerthick
    df = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    cap1 = cv.VideoCapture(inp1_path)
    cap2 = cv.VideoCapture(inp2_path)
    lengthoverlay1 = int(cap1.get(cv.CAP_PROP_FRAME_COUNT))
    length = int(cap2.get(cv.CAP_PROP_FRAME_COUNT))
    framerate = int(cap2.get(cv.CAP_PROP_FPS))
    #specifies the codec and creates a video writer object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(out_path, fourcc, framerate, (int(cap2.get(3)), \
                                                       int(cap2.get(4))))
    start1 = False
    start2 = False
    firstlines = {}
    secondlines = {}
    firstmin = df['frame'].min()
    diff = toi[0]-toi2[0]
    counter = 0
    for i in range(length):
        if(i == toi2[0]):
            start1 = True
            start2 = True
        if(i == toi2[0]+lengthoverlay1):
            start1 = False
        if(i == toi2[1]):
            start2 = False
        ret2, frame2 = cap2.read()
        if ret2 == True:
            if(start1):
                if(i-1+diff >= firstmin and tracers):
                    x1 = int(df.loc[df['frame'] == i-1+diff]\
                             ['x_center'].iloc[0])
                    y1 = int(df.loc[df['frame'] == i-1+diff]\
                             ['y_center'].iloc[0])
                    x2 = int(df.loc[df['frame'] == i+diff]['x_center'].iloc[0])
                    y2 = int(df.loc[df['frame'] == i+diff]['y_center'].iloc[0])
                    firstlines[i] = ((x1, y1), (x2, y2))
                ret1, frame1 = cap1.read()
                result = cv.bitwise_or(frame1, frame2)
            else:
                result = frame2
            if(start2):
                if(i-1 >= firstmin and tracers):
                    x1 = int(df2.loc[df2['frame'] == i-1]['x_center'].iloc[0])
                    y1 = int(df2.loc[df2['frame'] == i-1]['y_center'].iloc[0])
                    x2 = int(df2.loc[df2['frame'] == i]['x_center'].iloc[0])
                    y2 = int(df2.loc[df2['frame'] == i]['y_center'].iloc[0])
                    secondlines[i] = ((x1, y1), (x2, y2))
            if(len(firstlines) != 0 and len(secondlines) != 0 and \
               start1 == False and start2 == False):
                counter+=1
            if(counter == timelength):
                firstlines.clear()
                secondlines.clear()
            for line in firstlines:
                cv.line(result, firstlines[line][0], firstlines[line][1],
                        (0, 0, 255), thick)
            for line in secondlines:
                cv.line(result, secondlines[line][0], secondlines[line][1],
                        (255, 0, 0), thick)
            out.write(result)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap1.release()
    cap2.release()
    # Release the video writer object
    cv.destroyAllWindows()
