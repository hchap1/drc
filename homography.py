# homography.py
# Perspective warp to a top-down (bird's-eye) view.
#
# Usage:
#   from homography import birds_eye
#
#   warped = birds_eye(
#       frame,
#       front_right = ( 0.20, 0.10),   # metres right-of-centre,  metres ahead
#       front_left  = (-0.20, 0.10),
#       back_right  = ( 0.35, 0.60),
#       back_left   = (-0.35, 0.60),
#       pixels_per_unit = 400,          # 400 px per metre
#   )
#
# Coordinate convention
# ---------------------
#   x  →  right (positive = robot's right)
#   y  ↑  forward (positive = ahead of robot)
#
# The four corner arguments define where, on the ground plane, each
# corner of the camera's field of view falls.  The image is first
# flipped vertically (so "front/near" is at the top of the frame),
# then a perspective warp maps those four corners to their correct
# metric positions in the output image.
#
# In the output image:
#   +x (right)   → right
#   +y (forward) → upward   (standard overhead-map convention)

import cv2
import numpy as np


def birds_eye(
    image:            np.ndarray,
    front_right:      tuple[float, float],
    front_left:       tuple[float, float],
    back_right:       tuple[float, float],
    back_left:        tuple[float, float],
    pixels_per_unit:  float | None = None,
    output_size:      tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Return a top-down warped copy of `image`.

    Parameters
    ----------
    image            : BGR (or grayscale) numpy array from the camera.
    front_right      : (x, y) ground-plane coordinate visible at the
                       front-right corner of the camera's field of view.
    front_left       : same, front-left corner.
    back_right       : same, back-right corner  (farthest right).
    back_left        : same, back-left corner   (farthest left).
    pixels_per_unit  : output scale.  E.g. 400 means 400 px per ground
                       unit (metre, cm, …).  If None, the longer ground
                       axis is scaled to 400 px.
    output_size      : (width, height) in pixels.  Overrides
                       pixels_per_unit when supplied.

    Returns
    -------
    warped : np.ndarray with the same channel count as `image`.
    """

    # ------------------------------------------------------------------
    # 1. Flip vertically so the near/front field is at the TOP of the
    #    frame (makes the subsequent corner ordering intuitive: top-left
    #    == front-left, bottom-right == back-right, etc.).
    # ------------------------------------------------------------------
    flipped = cv2.flip(image, 0)
    h, w = flipped.shape[:2]

    # ------------------------------------------------------------------
    # 2. Source points: the four corners of the flipped image.
    #    Order matches the argument order (front-right, front-left,
    #    back-right, back-left).
    # ------------------------------------------------------------------
    src = np.float32([
        [w - 1, 0    ],   # front-right → top-right
        [0,     0    ],   # front-left  → top-left
        [w - 1, h - 1],   # back-right  → bottom-right
        [0,     h - 1],   # back-left   → bottom-left
    ])

    # ------------------------------------------------------------------
    # 3. Destination points: ground coordinates → output pixel positions.
    #    We find the bounding box of the four ground points and scale
    #    them uniformly so the output is metrically consistent.
    # ------------------------------------------------------------------
    gnd = np.array([front_right, front_left, back_right, back_left],
                   dtype=np.float32)

    min_x, min_y = gnd.min(axis=0)
    max_x, max_y = gnd.max(axis=0)
    span_x = max_x - min_x
    span_y = max_y - min_y

    if output_size is not None:
        out_w, out_h = output_size
        sx = out_w / span_x
        sy = out_h / span_y
    else:
        if pixels_per_unit is None:
            pixels_per_unit = 400.0 / max(span_x, span_y)
        sx = sy = float(pixels_per_unit)
        out_w = max(1, int(np.ceil(span_x * sx)))
        out_h = max(1, int(np.ceil(span_y * sy)))

    def _gnd_to_px(pt: tuple[float, float]) -> list[float]:
        gx, gy = pt
        # x: left→right in output matches +x in ground frame
        px = (gx - min_x) * sx
        # y: forward (+y) maps to TOP of output image (py = 0)
        py = (max_y - gy) * sy
        return [px, py]

    dst = np.float32([
        _gnd_to_px(front_right),
        _gnd_to_px(front_left),
        _gnd_to_px(back_right),
        _gnd_to_px(back_left),
    ])

    # ------------------------------------------------------------------
    # 4. Compute homography and warp.
    # ------------------------------------------------------------------
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(flipped, M, (out_w, out_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
    return warped


# ---------------------------------------------------------------------------
# Quick visual test:  python3 homography.py <image_path>
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        img = cv2.imread(path)
    else:
        # Synthetic test: a grey rectangle with a red dot at the centre
        img = np.full((480, 640, 3), 180, dtype=np.uint8)
        cv2.circle(img, (320, 240), 20, (0, 0, 255), -1)

    # Example calibration: camera sees a 40 cm wide × 50 cm deep patch
    # starting 10 cm in front of the robot.
    warped = birds_eye(
        img,
        front_right = ( 0.20, 0.10),
        front_left  = (-0.20, 0.10),
        back_right  = ( 0.20, 0.60),
        back_left   = (-0.20, 0.60),
        pixels_per_unit = 800,
    )

    cv2.imshow('original', img)
    cv2.imshow('birds_eye', warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
