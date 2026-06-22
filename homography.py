import cv2
import numpy as np


def birds_eye(
    flipped,
    front_right,
    front_left,
    back_right,
    back_left,
):
    h, w = flipped.shape[:2]

    src = np.float32([
        [w - 1, 0],
        [0, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ])

    gnd = np.array(
        [front_right, front_left, back_right, back_left],
        dtype=np.float32,
    )

    min_x, min_y = gnd.min(axis=0)
    max_x, max_y = gnd.max(axis=0)

    span_x = max_x - min_x
    span_y = max_y - min_y

    # Automatically choose a scale that gives reasonable resolution.
    scale = 1000.0 / max(span_x, span_y)

    out_w = max(1, int(np.ceil(span_x * scale)))
    out_h = max(1, int(np.ceil(span_y * scale)))

    def gnd_to_px(pt):
        gx, gy = pt

        px = (gx - min_x) * scale
        py = (max_y - gy) * scale

        return [px, py]

    dst = np.float32([
        gnd_to_px(front_right),
        gnd_to_px(front_left),
        gnd_to_px(back_right),
        gnd_to_px(back_left),
    ])

    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(
        flipped,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # ------------------------------------------------------------
    # Find the largest rectangle guaranteed to contain only valid
    # warped pixels.
    # ------------------------------------------------------------

    corners = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ]).reshape(-1, 1, 2)

    warped_corners = cv2.perspectiveTransform(corners, M).reshape(-1, 2)

    top = max(
        warped_corners[0, 1],
        warped_corners[1, 1],
    )

    bottom = min(
        warped_corners[2, 1],
        warped_corners[3, 1],
    )

    left = max(
        warped_corners[0, 0],
        warped_corners[3, 0],
    )

    right = min(
        warped_corners[1, 0],
        warped_corners[2, 0],
    )

    left = max(0, int(np.ceil(left)))
    right = min(out_w, int(np.floor(right)))

    top = max(0, int(np.ceil(top)))
    bottom = min(out_h, int(np.floor(bottom)))

    if right > left and bottom > top:
        warped = warped[top:bottom, left:right]

    return warped
