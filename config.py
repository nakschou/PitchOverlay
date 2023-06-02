class fileConfig:
    final_file_name = "coleoverlay" #Name of the final file, without .mp4
    pitch1_name = "colefb1" #Name of the file of the first pitch, without .mp4
    pitch1_velo = 98 #Velocity of the first pitch
    #Try and find this, otherwise set to a negative number
    #to have the program find it for you (it will not be 100% accurate)
    release1_frame = 180 
    pitch2_name = "colekc1" #Name of the file of the second pitch, without .mp4
    pitch2_velo = 86 #Velocity of the second pitch
    release2_frame = 182

    #Paths and suffixes
    processed_vids_path = "processed_vids/"
    pitcher_vids_path = "pitcher_vids/"
    csv_path = "csvs/"
    predictor_suffix = "_predictor"
    boxes_suffix = "_boxes"
    masks_suffix = "_masked"

    #Generally don't change these
    poly_deg = 3
    model_path = 'runs/detect/pitch_detection_v42/weights/best.pt'
    