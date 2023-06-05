import numpy as np

class fileConfig:

    # Please set these before using --------------------------------
    final_file_name = "kershawoverlay" #Name of the final file, without .mp4
    pitch1_name = "kershaw1" #Name of the file of the first pitch, without .mp4
    pitch1_velo = 91 #Velocity of the first pitch
    #Try and find this, otherwise set to a negative number
    #to have the program find it for you (it will not be 100% accurate)
    release1_frame = -1 
    pitch2_name = "kershaw2" #Name of the file of the second pitch, without .mp4
    pitch2_velo = 87 #Velocity of the second pitch
    release2_frame = -1
    line_hold_time = 0.5 #Time to hold the line on the screen
    tracers = True #Whether or not to draw tracers
    tracerthick = 2 #Thickness of the tracers

    #Paths and suffixes --------------------------------------
    processed_vids_path = "processed_vids/"
    pitcher_vids_path = "pitcher_vids/"
    csv_path = "csvs/"
    predictor_suffix = "_predictor"
    boxes_suffix = "_boxes"
    masks_suffix = "_masked"

    #Generally don't change these ----------------------------
    poly_deg = 3
    model_path = 'runs/detect/pitch_detection_v5/weights/best.pt'
    #Lower threshold for a pixel in hsv format
    pixel_low_thres = np.array([360, 255, 255])
    pixel_high_thres = np.array([360, 10, 255])
    box_size = 12
    