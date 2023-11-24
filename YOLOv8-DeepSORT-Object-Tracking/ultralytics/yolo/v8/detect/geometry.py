import json

import cv2
import numpy as np


def calc_speed(time: float, length_meters: int = 20):
    """
    Calculate the speed in kilometers per hour.

    Args:
    time (float): Time in seconds.
    length_meters (int): Length in meters, defaults to 20 meters.

    Returns:
    float: Speed in kilometers per hour.
    """
    if time <= 0:
        return 0

    # Convert meters per second to kilometers per hour
    speed_kmh = (length_meters / time) * (3600 / 1000)
    return speed_kmh

def get_area(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    area = merge_areas(data['areas'])
    return area

def merge_areas(areas):
    """ Merge multiple areas into a single area using cv2 for convex hull calculation.

    Args:
    areas (list): A list of areas, where each area is represented as a list of points.

    Returns:
    np.ndarray: An array of points representing the merged area.
    """
    # Combine all points from all areas into a single array
    all_points = np.vstack(areas).reshape(-1, 2)

    # Calculate the convex hull
    hull_indices = cv2.convexHull(np.array(all_points).astype(np.float32), returnPoints=False)
    merged_area = all_points[hull_indices.squeeze()]

    return merged_area
    
def is_point_in_area(point, area):
    """ Check if a point is inside a quadrilateral area using numpy for optimization.

    Args:
    point (tuple): A tuple representing the point (x, y).
    area (np.ndarray): A numpy array of points representing the area.

    Returns:
    bool: True if the point is inside the area, False otherwise.
    """
    x_coords, y_coords = area[:, 0], area[:, 1]

    n = len(area)
    inside = False

    for i in range(n):
        j = (i + 1) % n
        if ((y_coords[i] > point[1]) != (y_coords[j] > point[1])) and \
                (point[0] < (x_coords[j] - x_coords[i]) * (point[1] - y_coords[i]) / (y_coords[j] - y_coords[i]) + x_coords[i]):
            inside = not inside

    return inside
