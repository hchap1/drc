# hsv_tune.py
# Run on your laptop to tune HSV colour ranges live against the Jetson's camera.
# Connects to the raw frame stream (port 5008) from jetson_main_video.py.
# Shows a 4-panel view: raw frame, yellow mask, blue mask, combined detection.
# Press 'p' to print values ready to paste into process_cv.py.
# Press 'q' to quit.

import socket
import struct
import cv2
import numpy as np

JETSON_IP = '192.168.4.1'
PORT      = 5008

_HEADER = struct.Struct('<I')

# Match process_cv.py
PROC_W  = 160
PROC_H  = 90
ROI_TOP = 0.15

PANEL_SCALE = 3   # each panel is upscaled by this factor for visibility
MIN_PIXELS  = 40


def _recv_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    pos  = 0
    while pos < n:
        got = sock.recv_into(view[pos:], n - pos)
        if not got:
            raise ConnectionError('server closed the connection')
        pos += got
    return bytes(buf)


def _centroid(mask):
    col   = mask.sum(axis=0).astype(np.float32)
    total = col.sum()
    if total < MIN_PIXELS:
        return None
    xs = np.arange(len(col), dtype=np.float32)
    return float(np.dot(xs, col) / total)


def nothing(_):
    pass


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((JETSON_IP, PORT))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f'Connected to {JETSON_IP}:{PORT}  --  p=print values  q=quit')

    pw = PROC_W * PANEL_SCALE
    ph = PROC_H * PANEL_SCALE

    cv2.namedWindow('HSV Tune', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)

    def tb(name, default, maximum):
        cv2.createTrackbar(name, 'Controls', default, maximum, nothing)

    # Yellow sliders  (defaults from process_cv.py)
    tb('Y  H low',   8,   180)
    tb('Y  H high',  35,  180)
    tb('Y  S low',   80,  255)
    tb('Y  S high',  255, 255)
    tb('Y  V low',   80,  255)
    tb('Y  V high',  255, 255)

    # Blue sliders
    tb('B  H low',   90,  180)
    tb('B  H high',  130, 180)
    tb('B  S low',   60,  255)
    tb('B  S high',  255, 255)
    tb('B  V low',   30,  255)
    tb('B  V high',  255, 255)

    def get(name):
        return cv2.getTrackbarPos(name, 'Controls')

    try:
        while True:
            (length,) = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
            frame = cv2.imdecode(
                np.frombuffer(_recv_exact(sock, length), np.uint8),
                cv2.IMREAD_COLOR,
            )
            if frame is None:
                continue

            if frame.shape[1] != PROC_W or frame.shape[0] != PROC_H:
                frame = cv2.resize(frame, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)

            y0  = int(PROC_H * ROI_TOP)
            roi = frame[y0:]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            yl = np.array([get('Y  H low'),  get('Y  S low'),  get('Y  V low')])
            yh = np.array([get('Y  H high'), get('Y  S high'), get('Y  V high')])
            bl = np.array([get('B  H low'),  get('B  S low'),  get('B  V low')])
            bh = np.array([get('B  H high'), get('B  S high'), get('B  V high')])

            ym = cv2.inRange(hsv, yl, yh)
            bm = cv2.inRange(hsv, bl, bh)

            yx = _centroid(ym)
            bx = _centroid(bm)

            # ── Panel 1: raw frame with ROI line ──────────────────────────
            p_raw = frame.copy()
            cv2.line(p_raw, (0, y0), (PROC_W - 1, y0), (180, 180, 180), 1)

            # ── Panel 2: yellow mask ──────────────────────────────────────
            p_yellow = np.zeros_like(frame)
            p_yellow[y0:][ym > 0] = (0, 220, 220)
            if yx is not None:
                cv2.line(p_yellow, (int(yx), y0), (int(yx), PROC_H), (0, 255, 255), 1)

            # ── Panel 3: blue mask ────────────────────────────────────────
            p_blue = np.zeros_like(frame)
            p_blue[y0:][bm > 0] = (200, 80, 0)
            if bx is not None:
                cv2.line(p_blue, (int(bx), y0), (int(bx), PROC_H), (255, 80, 0), 1)

            # ── Panel 4: combined overlay ─────────────────────────────────
            p_combined = frame.copy()
            p_combined[y0:][ym > 0] = (0, 220, 220)
            p_combined[y0:][bm > 0] = (200, 80, 0)
            if yx is not None:
                cv2.line(p_combined, (int(yx), y0), (int(yx), PROC_H), (0, 255, 255), 1)
            if bx is not None:
                cv2.line(p_combined, (int(bx), y0), (int(bx), PROC_H), (255, 80, 0), 1)
            if yx is not None and bx is not None:
                mid = int((yx + bx) / 2)
                cv2.line(p_combined, (mid, y0), (mid, PROC_H), (0, 255, 0), 1)

            # ── Assemble 2×2 grid (upscaled for visibility) ───────────────
            def up(img):
                return cv2.resize(img, (pw, ph), interpolation=cv2.INTER_NEAREST)

            grid = np.vstack([
                np.hstack([up(p_raw),    up(p_yellow)]),
                np.hstack([up(p_blue),   up(p_combined)]),
            ])

            for text, x, y in [
                ('Raw + ROI',    4,      14),
                ('Yellow mask',  pw + 4, 14),
                ('Blue mask',    4,      ph + 14),
                ('Combined',     pw + 4, ph + 14),
            ]:
                cv2.putText(grid, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1)

            cv2.imshow('HSV Tune', grid)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('p'):
                print()
                print(f'YELLOW_LOW  = np.array([{yl[0]}, {yl[1]}, {yl[2]}])')
                print(f'YELLOW_HIGH = np.array([{yh[0]}, {yh[1]}, {yh[2]}])')
                print(f'BLUE_LOW    = np.array([{bl[0]}, {bl[1]}, {bl[2]}])')
                print(f'BLUE_HIGH   = np.array([{bh[0]}, {bh[1]}, {bh[2]}])')

    finally:
        sock.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
