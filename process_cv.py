import cv2
import numpy as np
from homography import birds_eye

def find_closest_blob_pixel(image, hsv_lower, hsv_upper):
    """
    Finds the pixel closest to the camera (bottom-centre of image)
    belonging to the closest blob matching the HSV threshold.

    Returns:
        (x, y) if a blob is found
        None otherwise
    """
    # Threshold
    # --------------------------------------------------
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        hsv_lower,
        hsv_upper,
    )

    # --------------------------------------------------
    # Morphological cleanup
    # --------------------------------------------------
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    # --------------------------------------------------
    # Connected components
    # --------------------------------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    h, w = mask.shape

    camera_x = w / 2
    camera_y = h

    best_blob = None
    best_dist_sq = float("inf")

    # --------------------------------------------------
    # Find closest blob (minimum distance from camera
    # to any pixel in the blob)
    # --------------------------------------------------
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area < 300:
            continue

        ys, xs = np.where(labels == label)

        dist_sq = (
            (xs - camera_x) ** 2 +
            (ys - camera_y) ** 2
        )

        closest_pixel_dist_sq = dist_sq.min()

        if closest_pixel_dist_sq < best_dist_sq:
            best_dist_sq = closest_pixel_dist_sq
            best_blob = label

    if best_blob is None:
        return mask, 0, 0

    # --------------------------------------------------
    # Find closest pixel within selected blob
    # --------------------------------------------------
    ys, xs = np.where(labels == best_blob)

    dist_sq = (
        (xs - camera_x) ** 2 +
        (ys - camera_y) ** 2
    )

    idx = np.argmin(dist_sq)

    return (
        mask,
        int(camera_x - xs[idx]),
        int(camera_y - ys[idx]),
    )

def process_frame(frame: np.ndarray):
    left, right = 0.0, 0.0

    blue_hsv_lower = np.array([80, 30, 20])
    blue_hsv_upper = np.array([150, 255, 255])
    yellow_hsv_lower = np.array([0, 60, 120])
    yellow_hsv_upper = np.array([70, 200, 255])

    image = birds_eye(
        frame,
        (85, 85),
        (-85, 85),
        (16, 10),
        (-16, 10),
    )

    blue_mask, blue_x, blue_y = find_closest_blob_pixel(image, blue_hsv_lower, blue_hsv_upper)

    yellow_mask, yellow_x, yellow_y = find_closest_blob_pixel(image, yellow_hsv_lower, yellow_hsv_upper)

    mask = cv2.bitwise_or(yellow_mask, blue_mask)

    if not (blue_x == 0 and blue_y == 0 and yellow_x == 0 and yellow_y == 0):
        y = abs(yellow_x)
        b = abs(blue_x)

        if b < y or y == 0:
            right = 0.3
            left = -0.2

        elif y <= b or b == 0:
            right = -0.2
            left = 0.3

    else:
        right = 0.1
        left = 0.1

    return left, right, mask

