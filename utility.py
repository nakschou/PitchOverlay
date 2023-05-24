def video_path(name, path):
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

def csv_path(name, path):
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

def video_path_suffix(name, path, suffix):
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

def csv_path_suffix(name, path, suffix):
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