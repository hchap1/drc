import cv2
import numpy as np
from homography import birds_eye

def process_frame(frame: np.ndarray):
    left, right = 0.0, 0.0

    image = birds_eye(
        frame,
        (1, 1),
        (0, 1),
        (1, 0),
        (0, 0),
        400
    )

    output_image = frame
    return left, right, output_image

