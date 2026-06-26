"""
cnn_backend.py — persistent CNN driving backend.

Spawned once by `run.py start`. Keeps model + camera hot indefinitely.
run.py connects and disconnects freely to arm/disarm motors without
restarting anything.

Usage (spawned automatically by run.py):
  python3 cnn_backend.py <model_folder>

Protocol (line-based over Unix socket):
  run.py → backend : START <speed_mult> <straight_mult> <launch_secs>
                     ARM
                     STOP
                     QUIT
  backend → run.py : LOG:<message>   (any time, during init or running)
                     READY            (sent once on connect, after init)
"""

import argparse
import os
import signal
import socket
import sys
import threading
import time

import cv2
import numpy as np
import torch

import motor_client
import video_server
from cnn_model import preprocess, IMG_W, IMG_H

SOCK_PATH     = '/tmp/drc_cnn.sock'
DEBUG_SKIP    = 2
FINISH_FRAMES = 4
SENSOR_W      = 1280
SENSOR_H      = 720
FRAMERATE     = 30
FLIP_METHOD   = 2

# ── Shared state ──────────────────────────────────────────────────────────────

_conn_lock    = threading.Lock()
_active_conn  = None          # current run.py connection, or None

_cfg_lock      = threading.Lock()
_speed_mult    = 1.0
_straight_mult = 1.0
_launch_time   = 0.0
_launch_mult   = 1.0   # boosted during launch ramp, returns to 1.0 after

_armed        = threading.Event()   # clear = disarmed
_initialized  = threading.Event()   # set once model + camera are ready

_latest_frame = None
_frame_lock   = threading.Lock()
_running      = True

motors    = None
debug_vid = None
_srv      = None


def _ramp_launch(duration):
    global _launch_mult
    ramp = min(0.5, duration)
    hold = duration - ramp
    steps = 30
    for i in range(1, steps + 1):
        with _cfg_lock:
            _launch_mult = 1.0 + (i / float(steps))   # 1.0 → 2.0
        time.sleep(ramp / steps)
    if hold > 0.0:
        time.sleep(hold)
    with _cfg_lock:
        _launch_mult = 1.0
    _log('[launch] boost off')


def _shutdown(signum, frame):
    global _running
    _running = False
    if _srv:
        _srv.close()
    if motors:
        motors.send(0.0, 0.0)
        motors.close()
    if debug_vid:
        debug_vid.close()
    try:
        os.unlink(SOCK_PATH)
    except FileNotFoundError:
        pass
    sys.exit(0)


def _log(msg):
    print(msg, flush=True)
    with _conn_lock:
        c = _active_conn
    if c is not None:
        try:
            c.sendall(f'LOG:{msg}\n'.encode())
        except OSError:
            pass


def _send(msg):
    with _conn_lock:
        c = _active_conn
    if c is not None:
        try:
            c.sendall(msg.encode())
        except OSError:
            pass


# ── GStreamer pipeline ────────────────────────────────────────────────────────

def _pipeline():
    return (
        f"nvarguscamerasrc sensor-id=0 ! "
        f"video/x-raw(memory:NVMM), width={SENSOR_W}, height={SENSOR_H}, framerate={FRAMERATE}/1 ! "
        f"nvvidconv flip-method={FLIP_METHOD} ! "
        f"video/x-raw(memory:NVMM), width={IMG_W}, height={IMG_H} ! "
        f"nvvidconv ! "
        f"video/x-raw, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


# ── Capture daemon ────────────────────────────────────────────────────────────

def _capture_loop(cap):
    global _latest_frame
    while _running:
        ok, frame = cap.read()
        if ok:
            with _frame_lock:
                _latest_frame = frame


# ── Finish-line detector ──────────────────────────────────────────────────────

_GREEN_LO      = np.array([40,  60,  60], dtype=np.uint8)
_GREEN_HI      = np.array([80, 255, 255], dtype=np.uint8)
_MIN_COL_FILL  = 0.55
_MIN_ROW_STACK = 3


def _finish_line_present(frame):
    roi       = frame[frame.shape[0] // 2:]
    hsv       = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask      = cv2.inRange(hsv, _GREEN_LO, _GREEN_HI)
    col_green = np.count_nonzero(mask, axis=0)
    wide_cols = np.count_nonzero(col_green >= _MIN_ROW_STACK)
    return wide_cols >= frame.shape[1] * _MIN_COL_FILL


# ── Connection handler ────────────────────────────────────────────────────────

def _handle_connection(conn):
    global _active_conn, _speed_mult, _straight_mult, _launch_time

    with _conn_lock:
        _active_conn = conn

    try:
        # Send READY as soon as init is done so launch knows it can exit
        _initialized.wait()
        _send('READY\n')

        for raw in conn.makefile('r'):
            cmd = raw.strip()

            if cmd.startswith('START'):
                parts = cmd.split()
                with _cfg_lock:
                    _speed_mult    = float(parts[1]) if len(parts) > 1 else 1.0
                    _straight_mult = float(parts[2]) if len(parts) > 2 else 1.0
                    _launch_time   = float(parts[3]) if len(parts) > 3 else 0.0
                _armed.clear()
                _log('[config] speed={:.3f}x  straight={:.3f}x  launch={:.2f}s'.format(
                    _speed_mult, _straight_mult, _launch_time))

            elif cmd == 'ARM':
                _armed.set()
                _log('[armed] motors enabled')
                with _cfg_lock:
                    launch = _launch_time
                if launch > 0.0:
                    _log('[launch] ramping CNN output 1x→2x over 0.5s for {:.2f}s total'.format(launch))
                    threading.Thread(target=_ramp_launch, args=(launch,), daemon=True).start()

            elif cmd == 'STOP':
                _armed.clear()
                motors.send(0.0, 0.0)
                _log('[stop] motors disarmed')
                break

            elif cmd == 'QUIT':
                _log('[quit] shutting down')
                _shutdown(None, None)

    except OSError:
        pass
    finally:
        with _conn_lock:
            _active_conn = None
        try:
            conn.close()
        except OSError:
            pass


def _server_loop(srv):
    while True:
        try:
            conn, _ = srv.accept()
        except OSError:
            break
        _handle_connection(conn)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _running, motors, debug_vid, _srv

    ap = argparse.ArgumentParser()
    ap.add_argument('folder', help='model folder containing model.pt')
    args = ap.parse_args()

    try:
        os.unlink(SOCK_PATH)
    except FileNotFoundError:
        pass

    _srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    _srv.bind(SOCK_PATH)
    _srv.listen(1)

    signal.signal(signal.SIGTERM, _shutdown)

    # Server thread starts immediately so run.py can connect during init
    # and see model-loading telemetry
    threading.Thread(target=_server_loop, args=(_srv,), daemon=True).start()

    path = os.path.join(args.folder, 'model.pt')
    if not os.path.exists(path):
        _log(f'[error] model not found: {path}')
        sys.exit(1)

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.jit.load(path, map_location=device_str).eval()
    _log(f'[model] TorchScript on {device_str}  ← {path}')
    device = torch.device(device_str)

    with torch.no_grad():
        dummy = torch.zeros(1, 3, IMG_H, IMG_W).to(device)
        model(dummy)
    _log('[model] warm-up done')

    cap = cv2.VideoCapture(_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        _log('[error] could not open CSI camera')
        sys.exit(1)
    threading.Thread(target=_capture_loop, args=(cap,), daemon=True).start()

    motors    = motor_client.connect()
    debug_vid = video_server.serve(port=5007, jpeg_quality=80, stream_width=None)
    _log(f'[video] debug feed on port 5007  ({IMG_W}×{IMG_H})')

    _initialized.set()
    _log('[backend] ready — waiting for run.py')

    frame_n      = 0
    finish_count = 0
    left         = 0.0
    right        = 0.0
    t0           = time.monotonic()
    t_log        = t0

    try:
        while _running:
            with _frame_lock:
                frame = _latest_frame

            if frame is None:
                continue

            if _finish_line_present(frame):
                finish_count += 1
                if finish_count >= FINISH_FRAMES and _armed.is_set():
                    _log('[finish] line detected — stopping')
                    motors.send(0.0, 0.0)
                    time.sleep(1.5)
                    _armed.clear()
                    finish_count = 0
            else:
                finish_count = 0

            try:
                with _cfg_lock:
                    sm  = _speed_mult
                    stm = _straight_mult

                tensor = preprocess(frame).to(device)
                with torch.no_grad():
                    out = model(tensor)
                left  = float(out[0, 0]) * sm
                right = float(out[0, 1]) * sm
                if abs(left - right) <= 0.08 * max(abs(left), abs(right)):
                    left  *= stm
                    right *= stm

                if _armed.is_set():
                    with _cfg_lock:
                        lm = _launch_mult
                    motors.send(
                        max(-1.0, min(1.0, left  * lm)),
                        max(-1.0, min(1.0, right * lm)),
                    )
                frame_n += 1

                if frame_n % DEBUG_SKIP == 0:
                    dbg = frame.copy()
                    cv2.putText(dbg, 'PT  L:{:+.3f}  R:{:+.3f}'.format(left, right),
                                (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    debug_vid.send(dbg)

                now = time.monotonic()
                if now - t_log >= 5.0:
                    _log('[fps] fps={:.1f}  L={:+.3f}  R={:+.3f}'.format(
                        frame_n / (now - t0), left, right))
                    t_log = now

            except Exception as e:
                _log('[error] inference error: {}'.format(e))

    finally:
        _running = False
        cap.release()
        motors.send(0.0, 0.0)
        motors.close()
        debug_vid.close()
        try:
            os.unlink(SOCK_PATH)
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    main()
