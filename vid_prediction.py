from ultralytics import YOLO
import numpy as np
import cv2 as cv
import pandas as pd

model_path = 'runs/detect/pitch_detection_v12/weights/best.pt'
vid_path = 'pitcher_vids/pitcher (1).mp4'
boxes_path = 'boxes.csv'
confidence_ind = 4
num_elems = 5

def get_boxes(model, vid_path):
    cap = cv.VideoCapture(vid_path)
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    boxes_dct = {}
    counter = 0
    while cap.isOpened():
        # read in frames
        _, image = cap.read()
        if image is None:
            break
        results = model.predict(source=image, save=True)
        # Create boxes dictionary
        no_boxes = len(results[0].boxes.xyxy)
        boxes_dct[counter] = {}
        for i in range(no_boxes):  # For every box
            # Gets length of coordinates array
            len_coords_arr = len(results[0].boxes.xyxy[i])
            # Creates a new array of coordinates with confidence
            coords_w_conf = np.zeros(len_coords_arr + 1)
            # Populates the new array
            for j in range(len_coords_arr):
                coords_w_conf[j] = results[0].boxes.xyxy[i][j]
            coords_w_conf[confidence_ind] = results[0].boxes.conf[i].item()
            boxes_dct[counter][f'box_{i+1}'] = coords_w_conf
        counter += 1
        if cv.waitKey(1) == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()
    return boxes_dct

def convert_boxes_dict(dct):
    return {(i, j): dct[i][j] for i in dct.keys() for j in dct[i].keys()}

def convert_boxes_df(dct):
    df = pd.DataFrame.from_dict(dct,
                                orient='index',
                                columns=['x1', 'y1', 'x2', 'y2', 'confidence'],
                                dtype=float)
    df.reset_index(inplace=True)
    return df

if __name__ == "__main__":
    # Loads the model, change pathing based on what you need
    model = YOLO(model_path)
    # Gets the boxes in a format unfit for a dataframe
    boxes_dct = get_boxes(model, vid_path)
    # Converts the boxes to fittable format and writes to dataframe
    boxes_dct = convert_boxes_dict(boxes_dct)
    df = convert_boxes_df(boxes_dct)
    # Saves the dataframe to a csv
    df.to_csv(boxes_path, index=False)

