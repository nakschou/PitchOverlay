import cv2 as cv
import config as cfg

def video_path(name: str, path: str) -> str:
    """
    Simple function to return the path of a video given the desired
    name and path.

    Args:
        name (str): Name of the video
        path (str): Path to the video
    
    Returns:
        str: Path to the video
    """
    return video_path_suffix(name, path, "")

def csv_path(name: str, path: str) -> str:
    """
    Simple function to return the path of a csv given the desired
    name and path.

    Args:
        name (str): Name of the video
        path (str): Path to the video
    
    Returns:
        str: Path to the video
    """
    return csv_path_suffix(name, path, "")

def video_path_suffix(name: str, path: str, suffix: str) -> str:
    """
    Simple function to return the path of an mp4 given the desired
    name and path.

    Args:
        name (str): Name of the video
        path (str): Path to the video
        suffix (str): Suffix of the video
    
    Returns:
        str: Path to the video
    """
    return path + name + suffix + ".mp4"

def csv_path_suffix(name: str, path: str, suffix: str) -> str:
    """
    Simple function to return the path of a csv given the desired
    name and path.

    Args:
        name (str): Name of the video
        path (str): Path to the video
        suffix (str): Suffix of the csv
    
    Returns:
        str: Path to the video
    """
    return path + name + suffix + ".csv"

def get_release_frame(vid_path: str) -> int:
    """
    Intended for the user to get the release frame of a video. This is
    the frame at which the video ends.

    Args:
        vid_path (str): Path to the video
    
    Returns:
        tuple (int, tuple): release frame, pixel location
    """
    cap = cv.VideoCapture(vid_path)
    framerate = int(cap.get(cv.CAP_PROP_FPS))
    current_frame = 0
    string = "'.' -> +1 frame"
    string2 = "',' -> -1 frame"
    string3 = "'s' -> save frame"
    string4 = "'r' -> reset to frame 0"
    string5 = "'-' -> -20 frames"
    string6 = "'=' -> +20 frames"
    str_arr = [string, string2, string3, string4, string5, string6]

    pixel_loc = None  # Default value assignment

    while cap.isOpened():
        # Set the frame position in the video
        cap.set(cv.CAP_PROP_POS_FRAMES, current_frame)
        # Read in frames
        _, image = cap.read()
        if image is None:
            break
        # Retrieve the current frame timestamp
        # print(cap.get(cv.CAP_PROP_POS_FRAMES))
        cv.rectangle(image, (25, 15), (230, 180), (0, 0, 0), -1)
        cv.putText(image, str(current_frame), (50, 50), 
                   cv.FONT_HERSHEY_SIMPLEX, 1,
                   (0, 255, 0), 2, cv.LINE_AA)
        i = 0
        for string in str_arr:
            cv.putText(image, string, (50, 75+i*15), cv.FONT_HERSHEY_SIMPLEX, 0.4, \
                          (0, 255, 255), 1, cv.LINE_AA) 
            i += 1
        # Display the current frame
        cv.imshow("Video", image)

        key = cv.waitKey(1)
        # Handle key presses
        if key == ord("."):
            current_frame += 1  # Move to the next frame
        elif key == ord(","):
            current_frame -= 1  # Move to the previous frame
        elif key == ord("s"):
            # Save color of a clicked pixel
            def mouse_callback(event, x, y, flags, param):
                nonlocal pixel_loc
                if event == cv.EVENT_LBUTTONDOWN:
                    pixel_loc = (x,y)

            cv.setMouseCallback("Video", mouse_callback)

            while pixel_loc is None:
                cv.waitKey(1)
            break
        elif key == ord("r"):
            current_frame = 0
        elif key == ord("-"):
            current_frame -= 20
        elif key == ord("="):
            current_frame += 20

    cap.release()
    cv.destroyAllWindows()

    if pixel_loc is None:
        pixel_loc = (0, 0)  # Assign default value if no mouse click was detected
    
    return (current_frame, pixel_loc)


if __name__ == "__main__":
    print(get_release_frame(video_path(cfg.fileConfig.pitch1_name, 
                                       cfg.fileConfig.pitcher_vids_path)))