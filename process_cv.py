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
ROI_TOP = 0.30   # use bottom 70% of frame for earlier line detection

MIN_PIXELS = 40

BASE_SPEED = 0.225   # 50% faster than original 0.15
MAX_SPEED  = 0.30

# Two gains: gentle when both lines visible (on-track / shallow wall approach),
# aggressive when only one line visible (corner following / exit).
STEER_GAIN_BOTH = 0.40
STEER_GAIN_ONE  = 0.75

# Desired x-position of each boundary line when it is the only one visible.
# Error goes to zero as the line returns to its normal lateral position
# → robot straightens naturally on corner exit without any state machine.
YELLOW_TARGET_X = PROC_W * 0.25
BLUE_TARGET_X   = PROC_W * 0.75

_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
_XS     = np.arange(PROC_W, dtype=np.float32)


def _col_centroid(mask):
    col   = mask.sum(axis=0).astype(np.float32)
    total = col.sum()
    if total < MIN_PIXELS:
        return None
    return float(np.dot(_XS, col) / total)


def _mix(error, gain):
    """Differential tank drive: outer motor at full speed, inner slows toward
    zero but never reverses.  Smooth swing near walls; tight pivot in corners."""
    steer = gain * error
    left  = min(MAX_SPEED, max(0.0, BASE_SPEED + steer))
    right = min(MAX_SPEED, max(0.0, BASE_SPEED - steer))
    return left, right


def process_frame(frame: np.ndarray):
    """Return (left, right, debug_image). Motor values are clamped to [0, MAX_SPEED]."""

    if frame.shape[1] != PROC_W or frame.shape[0] != PROC_H:
        frame = cv2.resize(frame, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)

    y0  = int(PROC_H * ROI_TOP)
    roi = frame[y0:]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    ym = cv2.morphologyEx(cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH), cv2.MORPH_OPEN, _KERNEL)
    bm = cv2.morphologyEx(cv2.inRange(hsv, BLUE_LOW,   BLUE_HIGH),   cv2.MORPH_OPEN, _KERNEL)

    yx = _col_centroid(ym)
    bx = _col_centroid(bm)

    cx = PROC_W / 2.0

    if yx is not None and bx is not None:
        # Both lines: gentle correction, no reverse — handles shallow wall approaches
        error = (((yx + bx) / 2.0) - cx) / cx
        left, right = _mix(error, STEER_GAIN_BOTH)
    elif yx is not None:
        # Only left line: aggressive tracking to follow corner and exit cleanly
        error = (yx - YELLOW_TARGET_X) / cx
        left, right = _mix(error, STEER_GAIN_ONE)
    elif bx is not None:
        # Only right line: same
        error = (bx - BLUE_TARGET_X) / cx
        left, right = _mix(error, STEER_GAIN_ONE)
    else:
        # Blind: creep straight
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
