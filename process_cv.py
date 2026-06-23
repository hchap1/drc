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
ROI_TOP = 0.30   # use bottom 70% of frame — looks further ahead for earlier reaction

MIN_PIXELS = 40  # minimum mask pixels to accept a colour as "found"

BASE_SPEED = 0.15
STEER_GAIN = 0.40
MAX_SPEED  = 0.20   # hard cap per competition rules

# Desired x-position for each line when it is the only one visible.
# Keeps the line at a fixed lateral position; as the corner straightens
# the line drifts back to this position and the error naturally goes to
# zero — the robot straightens without any explicit corner logic.
YELLOW_TARGET_X = PROC_W * 0.25   # yellow (left boundary) held at left quarter
BLUE_TARGET_X   = PROC_W * 0.75   # blue (right boundary) held at right quarter

_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
_XS     = np.arange(PROC_W, dtype=np.float32)   # pre-computed column indices


def _col_centroid(mask):
    """Column-weighted centroid of a binary mask. Returns None if too sparse."""
    col   = mask.sum(axis=0).astype(np.float32)
    total = col.sum()
    if total < MIN_PIXELS:
        return None
    return float(np.dot(_XS, col) / total)


def process_frame(frame: np.ndarray):
    """Return (left, right, debug_image). Motor values are clamped to ±MAX_SPEED."""

    if frame.shape[1] != PROC_W or frame.shape[0] != PROC_H:
        frame = cv2.resize(frame, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)

    # ── ROI: bottom portion only ──────────────────────────────────────────────
    y0  = int(PROC_H * ROI_TOP)
    roi = frame[y0:]

    # ── Colour detection (single HSV conversion, no MORPH_CLOSE at this res) ─
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    ym = cv2.morphologyEx(cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH), cv2.MORPH_OPEN, _KERNEL)
    bm = cv2.morphologyEx(cv2.inRange(hsv, BLUE_LOW,   BLUE_HIGH),   cv2.MORPH_OPEN, _KERNEL)

    yx = _col_centroid(ym)   # x-position of yellow (left) line
    bx = _col_centroid(bm)   # x-position of blue (right) line

    # ── Error computation ─────────────────────────────────────────────────────
    cx = PROC_W / 2.0

    if yx is not None and bx is not None:
        # Both lines: drive toward midpoint
        error = (((yx + bx) / 2.0) - cx) / cx
    elif yx is not None:
        # Only left line: hold it at its desired lateral position.
        # As the corner exit straightens the line returns to YELLOW_TARGET_X
        # → error→0 → robot straightens automatically.
        error = (yx - YELLOW_TARGET_X) / cx
    elif bx is not None:
        # Only right line: same principle.
        error = (bx - BLUE_TARGET_X) / cx
    else:
        speed = BASE_SPEED * 0.5
        return speed, speed, _debug(frame, y0, ym, bm, None, None, None, speed, speed)

    # ── Proportional controller ───────────────────────────────────────────────
    steer  = STEER_GAIN * error
    left   = max(-MAX_SPEED, min(MAX_SPEED, BASE_SPEED + steer))
    right  = max(-MAX_SPEED, min(MAX_SPEED, BASE_SPEED - steer))
    target = cx + error * cx   # recover display target from error

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
