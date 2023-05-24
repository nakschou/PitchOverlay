import cv2 as cv
import numpy as np
import os

def overlay_video(inp1_path, inp2_path, toi, out_path, csv1):
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
    for i in range(length):
        if(i == toi[0]):
            start = True
        if(i == toi[0]+lengthoverlay):
            start = False
        ret2, frame2 = cap2.read()
        if ret2 == True:
            if(start):
                ret1, frame1 = cap1.read()
                result = cv.addWeighted(frame1, 0.5, frame2, 0.5, 0)
            else:
                result = frame2
            out.write(result)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap1.release()
    cap2.release()
    # Release the video writer object
    cv.destroyAllWindows()
