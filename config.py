class fileConfig:
    final_file_name = "coleoverlay"
    pitch1_name = "colefb1"
    pitch2_name = "colekc1"
    pitch1_velo = 98
    pitch2_velo = 86
    release1_frame = -1
    release2_frame = -1
    processed_vids_path = "processed_vids/"
    pitcher_vids_path = "pitcher_vids/"
    csv_path = "csvs/"
    predictor_suffix = "_predictor"
    boxes_suffix = "_boxes"
    masks_suffix = "_masked"

    #Generally don't change these
    poly_deg = 3
    model_path = 'runs/detect/pitch_detection_v12/weights/best.pt'
    