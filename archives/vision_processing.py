import pandas as pd
import cv2 as cv
import numpy as np
import os

path = "pitcher_vids/pitcher (3).mp4"
boxes_path = 'new_boxes2.csv'
out_path = "tracker2processed.mp4"
poly_deg = 2

def get_circles(df, vid_path, out_path):
    """
    Creates a video with circles around the detected balls.

    Args:
        df (DataFrame): Dataframe containing bounding box data.
        vid_path (str): Path to video.
        out_path (str): Path to output video.
    
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
                else:
                    x1, y1, x2, y2 = 0, 0, 0, 0
                mask[y1:y2, x1:x2] = 255

                # Apply the mask to the input image
                masked_img = cv.bitwise_and(frame, frame, mask=mask)

                # Convert the masked image to grayscale
                gray = cv.cvtColor(masked_img, cv.COLOR_BGR2GRAY)

                # Apply Gaussian smoothing to the grayscale image
                gray_blur = cv.GaussianBlur(gray, (5, 5), 0)

                # Detect circles using the Hough transform
                circles = cv.HoughCircles(gray_blur, cv.HOUGH_GRADIENT, dp=1,
                                          minDist=2000, param1=50, param2=1, 
                                          minRadius=7, maxRadius=12)

                # Draw circles on the input image and adds to dict
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    for (x, y, r) in circles:
                        cv.circle(frame, (x, y), r, (0, 255, 0), 2)
                        circles_dct[i] = {'x': x, 'y': y, 'r': r}
                out.write(frame)
                cv.imshow('frame', frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break
    cap.release()
    # Release the video writer object
    out.release() 
    cv.destroyAllWindows()
    return circles_dct

def normalize_circles(df, dct):
    """
    Normalizes the coordinates of the circles to the bounding box.

    Args:
        df (DataFrame): Dataframe containing bounding box data.
        dct (dict): Dictionary containing circle coordinates.
    
    Returns:
        dataframe with new circles
    """
    df2 = pd.DataFrame(dct).transpose()
    df2.index.name = 'frame'
    df2.reset_index(inplace=True)
    print(df2)
    x_parametric = np.polyfit(df2['frame'], df2['x'], poly_deg)
    y_parametric = np.polyfit(df2['frame'], df2['y'], poly_deg)
    avg_r = df2['r'].mean()
    framemax = df2['frame'].max()
    framearr = np.zeros(framemax+1)
    for i in df2['frame']:
        framearr[i] = 1
    missing_rows = {}
    for i in range(len(framearr)):
        if(framearr[i] == 0):
            missing_rows[i] = [np.polyval(x_parametric, i),
                               np.polyval(y_parametric, i), avg_r]
    missing_df = pd.DataFrame.from_dict(missing_rows, orient='index', \
                                        columns=['x', 'y', 'r'])
    missing_df.index.name = 'frame'
    missing_df.reset_index(inplace=True)
    df2 = pd.concat([df2, missing_df], ignore_index=True)
    #sets dataframe's r to the average r
    df2['r'] = avg_r
    return df2

def save_vid(df, vid_path, out_path):
    """
    Creates a video with circles around the detected balls.

    Args:
        dct (dict): Dictionary containing circle data.
        vid_path (str): Path to video.
        out_path (str): Path to output video.
    
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
    for i in range(length):
        ret, frame = cap.read()
        if ret == True:
            if i in df['frame'].values:
                row = df.loc[df['frame'] == i]
                x = int(row['x'])
                y = int(row['y'])
                r = int(row['r'])
                cv.circle(frame, (x, y), r, (0, 255, 0), 2)
            out.write(frame)
            cv.imshow('frame', frame)
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
    dct = get_circles(df, path, out_path)
    #df2 = normalize_circles(df, dct)
    #print(df2)
    #save_vid(df2, path, out_path)