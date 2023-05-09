from ultralytics import YOLO
import numpy as np
import cv2 as cv
import json

model_path = 'runs/detect/pitch_detection_v12/weights/best.pt'
vid_path = 'pitcher_vids/pitcher (1).mp4'
boxes_path = 'boxes.txt'
confidence_ind = 5

def get_boxes(model, vid_path):
    cap = cv.VideoCapture(vid_path)
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    boxes_dct = {}
    counter = 0
    while cap.isOpened():
        #read in frames
        _, image = cap.read()
        if image is None:
            break
        results = model.predict(source=image, save=True)
        #Create boxes dictionary
        no_boxes = len(results[0].boxes.xyxy)
        boxes_dct[counter] = np.array(no_boxes)
        for i in range(no_boxes): #Populate boxes dictionary
            len_coords_arr = len(results[0].boxes.xyxy[i])
            coords_w_conf = np.array(len_coords_arr+1)
            for j in range(len_coords_arr):
                coords_w_conf[j] = results[0].boxes.xyxy[i][j]
            coords_w_conf[confidence_ind] = results[0].boxes.conf[i][0]
            boxes_dct[counter][i] = coords_w_conf
        counter += 1
        if cv.waitKey(1) == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()
    return boxes_dct

def dict_to_json(dict, file_loc):
    with open(file_loc, 'w') as convert_file:
     convert_file.write(json.dumps(dict))


if __name__ == "__main__":
    # Loads the model, change pathing based on what you need
    model = YOLO(model_path)
    boxes_dct = get_boxes(model, vid_path)
    dict_to_json(boxes_dct, boxes_path)
