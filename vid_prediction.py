from ultralytics import YOLO
import numpy as np
import cv2 as cv
import pandas as pd
import os

model_path = 'runs/detect/pitch_detection_v12/weights/best.pt'
vid_path = 'pitcher_vids/pitcher (3).mp4'
boxes_path = 'boxes2.csv'

confidence_ind = 4
num_extra_elems = 2
conf_threshold = 0.03

def get_boxes(model, vid_path):
    """
    Gets the boxes from the video and returns a dictionary of boxes
    
    Args:
        model: YOLO model
        vid_path: Path to video
    
    Returns:
        boxes_dct: Dictionary of boxes
    
    Raises:
        ValueError: If the path is invalid
    """
    if not os.path.isfile(vid_path):
        raise ValueError("Invalid path")
    cap = cv.VideoCapture(vid_path)
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    boxes_dct = {}
    counter = 0
    while cap.isOpened():
        # read in frames
        _, image = cap.read()
        if image is None:
            break
        results = model.predict(source=image, save=True, conf=conf_threshold)
        # Create boxes dictionary
        no_boxes = len(results[0].boxaes.xyxy)
        for i in range(no_boxes):  # For every box
            # Gets length of coordinates array
            len_coords_arr = len(results[0].boxes.xyxy[i])
            # Creates a new array of coordinates with confidence
            coords_w_conf = np.zeros(len_coords_arr + num_extra_elems)
            # Populates the new array (apparently xyxy are tensors)
            for j in range(len_coords_arr):
                coords_w_conf[j] = results[0].boxes.xyxy[i][j]
            coords_w_conf[confidence_ind] = results[0].boxes.conf[i].item()
            coords_w_conf[-1] = int(i+1)
            dct_key = (counter + 0.01*i)
            boxes_dct[dct_key] = coords_w_conf
        counter += 1
        if cv.waitKey(1) == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()
    return boxes_dct

def convert_boxes_df(dct):
    """
    Converts the boxes dictionary to a dataframe

    Args:
        dct: Dictionary of boxes
    
    Returns:
        df: Dataframe of boxes
    """
    df = pd.DataFrame.from_dict(dct, orient='index', columns=['x1', 'y1', 'x2',\
                                'y2', 'confidence', 'box_num'])
    #print(df)
    df.index.name = 'frame'
    df.reset_index(inplace=True)
    df = df.reindex(columns=['frame', 'box_num', 'x1', 'y1', 'x2', 'y2',
                            'confidence'])
    df['frame'] = df['frame'].astype(int)
    df['box_num'] = df['box_num'].astype(int)
    return df

if __name__ == "__main__":
    # Loads the model, change pathing based on what you need
    model = YOLO(model_path)
    # Tracks the video and saves it TODO: eliminate need for this
    model.track(vid_path, save=True, conf=conf_threshold)
    # Gets the boxes in a format unfit for a dataframe
    boxes_dct = get_boxes(model, vid_path)
    # Converts the boxes to fittable format and writes to dataframe
    df = convert_boxes_df(boxes_dct)
    # Saves the dataframe to a csv
    #print(df)
    df.to_csv(boxes_path, index=False)

