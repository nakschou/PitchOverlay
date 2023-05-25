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
        int: Release frame
    """
    cap = cv.VideoCapture(vid_path)
    counter = 0
    current_frame = 0
    string = "'.' -> +1 frame"
    string2 = "',' -> -1 frame"
    string3 = "'s' -> save frame"
    string4 = "'r' -> reset to frame 0"
    string5 = "'-' -> -20 frames"
    string6 = "'=' -> +20 frames"
    str_arr = [string, string2, string3, string4, string5, string6]
    while cap.isOpened():
        # read in frames
        _, image = cap.read()
        if image is None:
            break
        cv.rectangle(image, (25, 15), (230, 180), (0, 0, 0), -1)
        cv.putText(image, str(current_frame), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, \
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
            counter += 1
        elif key == ord(","):
            current_frame -= 1  # Move to the previous frame
            counter -= 1
        elif key == ord("s"):
            break
        elif key == ord("r"):
            current_frame = 0
            counter = 0
        elif key == ord("-"):
            current_frame -= 20
            counter -= 20
        elif key == ord("="):
            current_frame += 20
            counter += 20

        # Set the frame position in the video
        cap.set(cv.CAP_PROP_POS_FRAMES, current_frame)

    cap.release()
    cv.destroyAllWindows()
    return counter

if __name__ == "__main__":
    print(get_release_frame(video_path(cfg.fileConfig.pitch1_name, 
                                       cfg.fileConfig.pitcher_vids_path)))