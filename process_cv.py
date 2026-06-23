import cv2
import numpy as np

# ── Colour ranges (tune with hsv_tune.py) ────────────────────────────────────
YELLOW_LOW  = np.array([8,   80,  80])
YELLOW_HIGH = np.array([35, 255, 255])
BLUE_LOW    = np.array([90,  60,  30])
BLUE_HIGH   = np.array([130, 255, 255])

# ── Constants ─────────────────────────────────────────────────────────────────
PROC_W = 320
PROC_H = 180

MIN_AREA       = 200   # minimum blob pixels to consider a line real
HORIZ_THRESH   = 30    # line angle below this (deg from horizontal) = wall dead ahead

BASE_SPEED     = 0.15
MAX_SPEED      = 0.20
STEER_GAIN     = 0.40  # normal on-track gain
CORNER_GAIN    = 0.60  # gain when corner-locked (more aggressive)

# Frames of single-line before locking to outside wall
CORNER_IN      = 8
# Frames of both-lines needed to unlock
CORNER_OUT     = 20
# Offset from outside wall when corner-locked (fraction of frame width)
CORNER_OFFSET  = 0.35

_single_ctr  = 0
_both_ctr    = 0
_locked      = False
_locked_col  = None   # 'yellow' or 'blue'

_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def _largest_blob(mask):
    """
    Find largest blob in mask.
    Returns (centroid_x, angle_from_horizontal_deg, area) or None.
    angle_from_horizontal: 0 = horizontal line, 90 = vertical line.
    """
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return None

    idx  = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    area = int(stats[idx, cv2.CC_STAT_AREA])
    if area < MIN_AREA:
        return None

    cx = float(centroids[idx, 0])

    # Fit a line to the blob pixels to get its orientation
    ys, xs = np.where(labels == idx)
    pts    = np.column_stack([xs, ys]).astype(np.float32).reshape(-1, 1, 2)
    vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    # arctan2(|vy|, |vx|): 0 = horizontal direction, 90 = vertical direction
    angle = float(np.degrees(np.arctan2(abs(float(vy[0])), abs(float(vx[0])))))

    return cx, angle, area


def _steer(error, gain=STEER_GAIN):
    s     = gain * error
    left  = max(-MAX_SPEED, min(MAX_SPEED, BASE_SPEED + s))
    right = max(-MAX_SPEED, min(MAX_SPEED, BASE_SPEED - s))
    return left, right


def process_frame(frame: np.ndarray):
    global _single_ctr, _both_ctr, _locked, _locked_col

    if frame.shape[1] != PROC_W or frame.shape[0] != PROC_H:
        frame = cv2.resize(frame, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)

    # Full frame — no ROI crop
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    ym = cv2.inRange(hsv, YELLOW_LOW,  YELLOW_HIGH)
    bm = cv2.inRange(hsv, BLUE_LOW,    BLUE_HIGH)
    ym = cv2.morphologyEx(cv2.morphologyEx(ym, cv2.MORPH_OPEN,  _KERNEL), cv2.MORPH_CLOSE, _KERNEL)
    bm = cv2.morphologyEx(cv2.morphologyEx(bm, cv2.MORPH_OPEN,  _KERNEL), cv2.MORPH_CLOSE, _KERNEL)

    y = _largest_blob(ym)   # (cx, angle, area) or None
    b = _largest_blob(bm)

    cx = PROC_W / 2.0

    # ── Emergency: wall dead ahead ────────────────────────────────────────────
    # A nearly-horizontal line means the robot is heading straight into that wall.
    if y is not None and y[1] < HORIZ_THRESH:
        l, r = _steer(1.0, CORNER_GAIN)   # yellow wall → hard right
        return l, r, _debug(frame, ym, bm, y, b, None, l, r, 'Y-WALL')

    if b is not None and b[1] < HORIZ_THRESH:
        l, r = _steer(-1.0, CORNER_GAIN)  # blue wall → hard left
        return l, r, _debug(frame, ym, bm, y, b, None, l, r, 'B-WALL')

    # ── Corner lock state machine ─────────────────────────────────────────────
    both = y is not None and b is not None

    if both:
        _single_ctr = 0
        _both_ctr  += 1
        if _locked and _both_ctr >= CORNER_OUT:
            _locked     = False
            _locked_col = None
    else:
        _both_ctr = 0
        if not _locked:
            _single_ctr += 1
            if _single_ctr >= CORNER_IN:
                _locked     = True
                _locked_col = 'yellow' if y is not None else 'blue'

    # ── Motor decisions ───────────────────────────────────────────────────────
    target = None

    if _locked:
        mode = f'COR:{_locked_col[0].upper()}'
        info = y if _locked_col == 'yellow' else b

        if info is None:
            # Lost the locked wall — keep creeping forward
            return 0.1, 0.1, _debug(frame, ym, bm, y, b, None, 0.1, 0.1, 'COR:BLIND')

        # Keep a fixed distance to the outside wall
        off    = PROC_W * CORNER_OFFSET
        target = info[0] + (off if _locked_col == 'yellow' else -off)
        l, r   = _steer((target - cx) / cx, CORNER_GAIN)

    elif both:
        mode   = 'TRACK'
        target = (y[0] + b[0]) / 2.0
        l, r   = _steer((target - cx) / cx)

    elif y is not None:
        mode   = 'Y-ONLY'
        target = y[0] + PROC_W * 0.30
        l, r   = _steer((target - cx) / cx)

    elif b is not None:
        mode   = 'B-ONLY'
        target = b[0] - PROC_W * 0.30
        l, r   = _steer((target - cx) / cx)

    else:
        return 0.1, 0.1, _debug(frame, ym, bm, None, None, None, 0.1, 0.1, 'BLIND')

    return l, r, _debug(frame, ym, bm, y, b, target, l, r, mode)


def _debug(frame, ym, bm, y, b, target, left, right, mode):
    out = frame.copy()
    h, w = out.shape[:2]

    out[ym > 0] = (0, 220, 220)
    out[bm > 0] = (200, 80, 0)

    if y is not None:
        xi = int(y[0])
        cv2.line(out, (xi, 0), (xi, h), (0, 255, 255), 1)
        # draw angle indicator (line through centroid at detected angle)
        ang = np.radians(y[1])
        dx, dy = int(40 * np.cos(ang)), int(40 * np.sin(ang))
        cv2.line(out, (xi - dx, h//2 + dy), (xi + dx, h//2 - dy), (0, 255, 255), 2)

    if b is not None:
        xi = int(b[0])
        cv2.line(out, (xi, 0), (xi, h), (255, 80, 0), 1)
        ang = np.radians(b[1])
        dx, dy = int(40 * np.cos(ang)), int(40 * np.sin(ang))
        cv2.line(out, (xi - dx, h//2 + dy), (xi + dx, h//2 - dy), (255, 80, 0), 2)

    if target is not None:
        cv2.line(out, (int(target), 0), (int(target), h), (0, 255, 0), 1)

    cv2.putText(out, f'{mode} L:{left:+.2f} R:{right:+.2f}', (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out
