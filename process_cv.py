import cv2
import numpy as np

# ── Colour ranges (tune with hsv_tune.py) ────────────────────────────────────
YELLOW_LOW  = np.array([8,   80,  80])
YELLOW_HIGH = np.array([35, 255, 255])
BLUE_LOW    = np.array([90,  60,  30])
BLUE_HIGH   = np.array([130, 255, 255])

# ── Processing constants ──────────────────────────────────────────────────────
PROC_W     = 320
PROC_H     = 180
ROI_TOP    = 0.45   # ignore top 45% of frame; focus on near-ground region

MIN_PIXELS = 150    # minimum mask pixels to accept a colour as "found"

BASE_SPEED        = 0.15
CORNER_SPEED      = BASE_SPEED * 0.6   # slower when only one line visible
STEER_GAIN        = 0.40
MAX_SPEED         = 0.20               # hard cap per competition rules
CORNER_OFFSET     = 0.50               # fraction of frame width to offset target when one line lost

_KERNEL     = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
_last_steer = 0.0                      # remembered steer for blind-corner recovery


def _col_centroid(mask):
    """Column-weighted centroid of a binary mask. Returns None if too sparse."""
    col   = mask.sum(axis=0).astype(np.float32)
    total = col.sum()
    if total < MIN_PIXELS:
        return None
    xs = np.arange(len(col), dtype=np.float32)
    return float(np.dot(xs, col) / total)


def process_frame(frame: np.ndarray):
    """Return (left, right, debug_image). Motor values are clamped to ±MAX_SPEED."""
    global _last_steer

    # Guard: ensure consistent resolution regardless of capture source
    if frame.shape[1] != PROC_W or frame.shape[0] != PROC_H:
        frame = cv2.resize(frame, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)

    # ── ROI: bottom portion only ──────────────────────────────────────────────
    y0  = int(PROC_H * ROI_TOP)
    roi = frame[y0:]

    # ── Colour detection (single HSV conversion) ──────────────────────────────
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    ym = cv2.inRange(hsv, YELLOW_LOW,  YELLOW_HIGH)
    bm = cv2.inRange(hsv, BLUE_LOW,    BLUE_HIGH)

    ym = cv2.morphologyEx(ym, cv2.MORPH_OPEN,  _KERNEL)
    ym = cv2.morphologyEx(ym, cv2.MORPH_CLOSE, _KERNEL)
    bm = cv2.morphologyEx(bm, cv2.MORPH_OPEN,  _KERNEL)
    bm = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, _KERNEL)

    yx = _col_centroid(ym)   # x-position of yellow (left) line
    bx = _col_centroid(bm)   # x-position of blue (right) line

    cx = PROC_W / 2.0

    # ── Steering target ───────────────────────────────────────────────────────
    if yx is not None and bx is not None:
        target = (yx + bx) / 2.0
        fwd = BASE_SPEED
    elif yx is not None:
        # Only left line: large offset steers hard right into the missing inside
        target = yx + PROC_W * CORNER_OFFSET
        fwd = CORNER_SPEED
    elif bx is not None:
        # Only right line: large offset steers hard left into the missing inside
        target = bx - PROC_W * CORNER_OFFSET
        fwd = CORNER_SPEED
    else:
        # No lines at all: commit to the last known steer so the robot keeps
        # turning through the hairpin rather than going straight and losing track
        left  = max(-MAX_SPEED, min(MAX_SPEED, CORNER_SPEED + _last_steer))
        right = max(-MAX_SPEED, min(MAX_SPEED, CORNER_SPEED - _last_steer))
        return left, right, _debug(frame, y0, ym, bm, None, None, None, left, right)

    # ── Proportional controller ───────────────────────────────────────────────
    # error > 0: target is right of centre → turn right (left motor > right motor)
    error = (target - cx) / cx
    steer = STEER_GAIN * error
    _last_steer = steer   # save for blind-corner recovery
    left  = max(-MAX_SPEED, min(MAX_SPEED, fwd + steer))
    right = max(-MAX_SPEED, min(MAX_SPEED, fwd - steer))

    n_lines = (yx is not None) + (bx is not None)
    return left, right, _debug(frame, y0, ym, bm, yx, bx, target, left, right, n_lines)


def _debug(frame, y0, ym, bm, yx, bx, target, left, right, n_lines=2):
    out = frame.copy()
    h   = frame.shape[0]

    out[y0:][ym > 0] = (0, 220, 220)   # yellow line overlay
    out[y0:][bm > 0] = (200, 80, 0)    # blue line overlay

    if yx is not None:
        cv2.line(out, (int(yx),     y0), (int(yx),     h), (0, 255, 255), 1)
    if bx is not None:
        cv2.line(out, (int(bx),     y0), (int(bx),     h), (255, 80,  0), 1)
    if target is not None:
        cv2.line(out, (int(target), y0), (int(target), h), (0, 255,   0), 1)

    mode = ['BLIND', 'CORNER', 'TRACK'][n_lines]
    cv2.putText(out, f'{mode} L:{left:+.2f} R:{right:+.2f}', (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return out
