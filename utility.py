import cv2 as cv
import config as cfg
import math

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

def pitch_time_frames(speed: int) -> int:
    """
    Calculates the time it takes for a pitch to reach the plate.

    Args:
        speed: The speed of the pitch in mph
    
    Returns:
        The time it takes for the pitch to reach the plate in frames
    """
    # Convert mph to m/s
    v0 = speed * 0.44704
    # Adjust for release angle
    v0 = v0 * math.cos(math.radians(math.pi/36))

    # Constants
    c_d = 0.47      # Drag coefficient of a sphere
    a = 0.004145    # Cross-sectional area of a baseball (m^2)
    m = 0.145       # Mass of a baseball (kg)
    rho = 1.225     # Density of air at room temperature (kg/m^3)
    dist = 18.44 + 1.28 # Distance from pitcher to plate (m)
    # the weirdo number is to account for the fact that the catcher generally
    # catches the ball a little bit behind the plate

    # Calculate the drag force
    f_d = 0.5 * c_d * a * rho * v0**2

    # Calculate the acceleration due to air resistance
    a_D = f_d / m

    # Calculate the time to reach the plate
    t = (-v0 + math.sqrt(v0**2 + 2*a_D*dist)) / a_D

    return int(t*60+0.5)

if __name__ == "__main__":
    print(get_release_frame(video_path(cfg.fileConfig.pitch1_name, 
                                       cfg.fileConfig.pitcher_vids_path)))