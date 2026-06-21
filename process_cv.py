import cv2
import numpy as np

def process_frame(frame: np.ndarray):
    """
    *** Fill this in with your CV logic. ***

    frame -- raw BGR image straight off the camera (np.ndarray)

    Must return (left, right, output_image):
        left, right   -- motor powers in [-1.0, 1.0]
        output_image  -- BGR np.ndarray (e.g. `frame` with overlays drawn
                          on it) -- this is what gets streamed
    """
    left, right = 0.0, 0.0
    output_image = frame
    return left, right, output_image

