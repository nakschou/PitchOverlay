import pandas as pd
import cv2 as cv
import numpy as np
import os

path = "pitcher_vids/pitcher (3).mp4"
boxes_path = 'new_boxes2.csv'
out_path = "tracker2processed.mp4"
poly_deg = 2

def get_circles(df, vid_path):
    """
    Creates a video with circles around the detected balls.

    Args:
        df (DataFrame): Dataframe containing bounding box data.
        vid_path (str): Path to video.
    
    Returns:
        None
    
    Raises:
        ValueError: If vid_path is not a valid path.
    """
    if not os.path.isfile(path):
        raise ValueError("Invalid path")
    cap = cv.VideoCapture(vid_path)
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    circles_dct = {}
    for i in range(length):
        ret, frame = cap.read()
        if ret == True:
            # Define the region of interest (ROI) as a rectangular mask
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                if i in df['frame'].values:
                    row = df.loc[df['frame'] == i]
                    x1 = int(row['x1'].iloc[0])
                    y1 = int(row['y1'].iloc[0])
                    x2 = int(row['x2'].iloc[0])
                    y2 = int(row['y2'].iloc[0])
                    x_center = int(row['x_center'].iloc[0])
                    y_center = int(row['y_center'].iloc[0])
                    mask[y1:y2, x1:x2] = 255
                    # Apply the mask to the input image
                    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                    masked_img = cv.bitwise_and(hsv, hsv, mask=mask)

                    # Get the pixel values at (x, y)
                    pixel = hsv[y_center, x_center]
        
                    lower = pixel - np.array([360, 15, 70])
                    upper = pixel + np.array([360, 15, 70])
                
                    # preparing the mask to overlay
                    mask = cv.inRange(hsv, lower, upper)
                    
                    result = cv.bitwise_and(masked_img, frame, mask = mask)
                    non_zero_pixels = np.where(result != 0)
                    #print(non_zero_pixels)

                    # Calculate data for non-zero pixels
                    mean_x = int(np.mean(non_zero_pixels[1]))
                    mean_y = int(np.mean(non_zero_pixels[0]))
                    mean_dist = int(np.mean(np.sqrt(np.square(non_zero_pixels\
                                        [1] - mean_x) +
                                        np.square(non_zero_pixels[0] - 
                                        mean_y))))

                    circles_dct[i] = (mean_x, mean_y, 2*mean_dist)
                    cv.circle(frame, (mean_x, mean_y), 8, 
                              (0, 255, 0), 2)
                else:
                    mask = frame
                    result = frame
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break
    cap.release()
    # Release the video writer object
    cv.destroyAllWindows()
    return circles_dct

def get_mask(dct, vid_path, out_path):
    """
    Creates a video with circles around the detected balls.

    Args:
        dct (dict): Dictionary containing circle data.
        vid_path (str): Path to video.
        out_path (str): Path to save video.
    
    Returns:
        None
    
    Raises:
        ValueError: If vid_path is not a valid path.
    """
    if not os.path.isfile(path):
        raise ValueError("Invalid path")
    cap = cv.VideoCapture(vid_path)
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    framerate = int(cap.get(cv.CAP_PROP_FPS))
    #specifies the codec and creates a video writer object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(out_path, fourcc, framerate, (int(cap.get(3)), \
                                                       int(cap.get(4))))
    avg_radius = int(np.mean([i[2] for i in dct.values()]))
    for i in range(length):
        ret, frame = cap.read()
        if ret == True:
            if i in dct.keys():
                mask = np.zeros((int(cap.get(4)), int(cap.get(3)), 3), np.uint8)
                x, y = dct[i][0], dct[i][1]
                # Define the circle
                center = (x, y)
                radius = avg_radius
                color = (255, 255, 255)

                # Draw the defined circle on the mask
                cv.circle(mask, center, radius, color, -1)

                # Apply the mask to the original image
                result = cv.bitwise_and(frame, mask)
                out.write(result)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    # Release the video writer object
    out.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    df = pd.read_csv(boxes_path)
    dct = get_circles(df, path)
    get_mask(dct, path, out_path)
    #df2 = normalize_circles(df, dct)
    #print(df2)
    #save_vid(df2, path, out_path)