import cv2
import numpy as np

# ── Colour ranges (tune with hsv_tune.py) ────────────────────────────────────
YELLOW_LOW  = np.array([8,   80,  80])
YELLOW_HIGH = np.array([35, 255, 255])
BLUE_LOW    = np.array([90,  60,  30])
BLUE_HIGH   = np.array([130, 255, 255])

# ── Processing constants ──────────────────────────────────────────────────────
PROC_W  = 160
PROC_H  = 90

# Drop only the top 15% of the frame (avoids overhead distractions while
# still giving ~85% vertical coverage for early corner detection).
ROI_TOP = 0.15

MIN_PIXELS = 30   # raw pixel count before weighting

BASE_SPEED      = 0.225
MAX_SPEED       = 0.30
STEER_GAIN_BOTH = 0.40   # gentle on-track / shallow wall correction
STEER_GAIN_ONE  = 0.80   # aggressive corner following and exit

# When only one boundary line is visible, hold it at these x-positions.
# Pushing them toward the edges keeps the robot well away from the wall
# it can see. Error→0 as the line returns here on corner exit.
YELLOW_TARGET_X = PROC_W * 0.18   # left boundary kept far to the left
BLUE_TARGET_X   = PROC_W * 0.82   # right boundary kept far to the right

_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
_XS     = np.arange(PROC_W, dtype=np.float32)

# Far-field row weights: top rows of the ROI (furthest ahead) get 2.5×,
# bottom rows (close to robot) get 0.5×. Re-computed if ROI height changes.
_row_weights = None


def _get_weights(h):
    global _row_weights
    if _row_weights is None or _row_weights.shape[0] != h:
        _row_weights = np.linspace(2.5, 0.5, h, dtype=np.float32).reshape(-1, 1)
    return _row_weights


def _col_centroid(mask, weights):
    """Far-weighted column centroid. Far rows steer earlier into corners."""
    if mask.sum() < MIN_PIXELS:
        return None
    col   = (mask.astype(np.float32) * weights).sum(axis=0)
    total = col.sum()
    return float(np.dot(_XS, col) / total) if total > 0 else None


def _mix(error, gain):
    """Differential tank drive: outer motor at speed, inner toward 0, never reverse."""
    steer = gain * error
    left  = min(MAX_SPEED, max(0.0, BASE_SPEED + steer))
    right = min(MAX_SPEED, max(0.0, BASE_SPEED - steer))
    return left, right


def process_frame(frame: np.ndarray):
    """Return (left, right, debug_image). Motor values clamped to [0, MAX_SPEED]."""

    if frame.shape[1] != PROC_W or frame.shape[0] != PROC_H:
        frame = cv2.resize(frame, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)

    y0  = int(PROC_H * ROI_TOP)
    roi = frame[y0:]
    w   = _get_weights(roi.shape[0])

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    ym  = cv2.morphologyEx(cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH), cv2.MORPH_OPEN, _KERNEL)
    bm  = cv2.morphologyEx(cv2.inRange(hsv, BLUE_LOW,   BLUE_HIGH),   cv2.MORPH_OPEN, _KERNEL)

    yx = _col_centroid(ym, w)
    bx = _col_centroid(bm, w)

    cx = PROC_W / 2.0

    if yx is not None and bx is not None:
        # Both lines: steer toward midpoint with gentle gain.
        # No reverse — smooth differential correction near walls.
        error = (((yx + bx) / 2.0) - cx) / cx
        left, right = _mix(error, STEER_GAIN_BOTH)

    elif yx is not None:
        # Only left (yellow) line: track it firmly at YELLOW_TARGET_X.
        # Far-weighting means straightening is detected early on corner exit.
        error = (yx - YELLOW_TARGET_X) / cx
        left, right = _mix(error, STEER_GAIN_ONE)

    elif bx is not None:
        # Only right (blue) line: same principle.
        error = (bx - BLUE_TARGET_X) / cx
        left, right = _mix(error, STEER_GAIN_ONE)

    else:
        # Blind: creep straight until a line reappears.
        left = right = BASE_SPEED * 0.5
        error = 0.0

    target = cx + error * cx
    return left, right, _debug(frame, y0, ym, bm, yx, bx, target, left, right)


def _debug(frame, y0, ym, bm, yx, bx, target, left, right):
    out = frame.copy()
    h   = frame.shape[0]

    out[y0:][ym > 0] = (0, 220, 220)
    out[y0:][bm > 0] = (200, 80, 0)

    if yx is not None:
        cv2.line(out, (int(yx),     y0), (int(yx),     h), (0, 255, 255), 1)
    if bx is not None:
        cv2.line(out, (int(bx),     y0), (int(bx),     h), (255, 80,  0), 1)
    if target is not None:
        cv2.line(out, (int(target), y0), (int(target), h), (0, 255,   0), 1)

    cv2.putText(out, f'L:{left:+.2f} R:{right:+.2f}', (2, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    return out
