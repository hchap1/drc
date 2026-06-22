import cv2
import numpy as np
from homography import birds_eye

def process_frame(frame: np.ndarray):
    left, right = 0.0, 0.0

    image = birds_eye(
        frame,
        (85, 85),
        (-85, 85),
        (16, 10),
        (-16, 10),
    )

    output_image = image
    return left, right, output_image

