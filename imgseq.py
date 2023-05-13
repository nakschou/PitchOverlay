import cv2 as cv
import os

vid_path = 'pitcher_vids/cut up/cutup.mp4'
seq_path = 'sequence/'

def create_sequence(vid_path, seq_path):
    """
    Creates a sequence of images from a video
    
    Args:
        vid_path: Path to video
        seq_path: Path to save sequence to
    
    Returns:
        None
    
    Raises:
        ValueError: If the path is invalid
    """
    if not os.path.isfile(vid_path):
        raise ValueError("Invalid path")
    cap = cv.VideoCapture(vid_path)
    counter = 0
    while cap.isOpened():
        # read in frames
        _, image = cap.read()
        if image is None:
            break
        cv.imwrite(seq_path + str(counter) + '.jpg', image)
        counter += 1
        if cv.waitKey(1) == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    create_sequence(vid_path, seq_path)
