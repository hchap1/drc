"""
jetson_collect.py — Jetson-side data collection server.

The laptop runs collect_data.py which drives the robot and toggles recording.
This script receives motor + recording packets over TCP, forwards them to the
ESP32, and saves every camera frame + the motor values current at capture time.

Controls are on the LAPTOP (collect_data.py):
  W/A/S/D        drive
  R              toggle recording
  Q / ESC        quit

Launch on Jetson (survives SSH disconnect):
  nohup python3 jetson_collect.py > ~/collect.log 2>&1 &

Stop:
  kill $(pgrep -f jetson_collect.py)
"""

import csv
import queue
import signal
import socket
import struct
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

import motor_client
from cnn_model import IMG_W, IMG_H

# ── Network ─────────────────────────────────────────────────────────────────────
LABEL_PORT = 5009
_PKT       = struct.Struct('<Bff')   # recording(uint8), left(float32), right(float32)

# ── Camera ──────────────────────────────────────────────────────────────────────
JPEG_Q      = 90
SENSOR_W    = 1280
SENSOR_H    = 720
FRAMERATE   = 30
FLIP_METHOD = 2

# ── Shared motor/recording state (written by TCP thread, read by capture thread) ─
_recording = False
_left      = 0.0
_right     = 0.0
_state_lock = threading.Lock()

_running = True


# ── GStreamer pipeline ───────────────────────────────────────────────────────────

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


# ── Capture thread — saves every frame at camera rate ───────────────────────────

def _capture_loop(cap, save_q, frame_idx_ref):
    t_print = time.monotonic()
    while _running:
        ok, frame = cap.read()
        if not ok:
            continue

        with _state_lock:
            rec   = _recording
            left  = _left
            right = _right

        if rec:
            fname = f'{frame_idx_ref[0]:06d}.jpg'
            save_q.put((fname, frame.copy(), left, right))  # blocks if queue full
            frame_idx_ref[0] += 1

            now = time.monotonic()
            if now - t_print >= 1.0:
                print(f'[rec] {frame_idx_ref[0]} frames', flush=True)
                t_print = now


# ── Save thread ──────────────────────────────────────────────────────────────────

def _flush(csv_path, rows):
    if not rows:
        return
    write_header = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['frame', 'left', 'right'])
        w.writerows(rows)
    rows.clear()


def _save_loop(save_q, fdir, csv_path):
    rows = []
    while True:
        item = save_q.get()
        if item is None:
            _flush(csv_path, rows)
            save_q.task_done()
            break
        fname, frame, left, right = item
        ok, enc = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        if ok:
            (fdir / fname).write_bytes(enc.tobytes())
            rows.append((fname, round(left, 4), round(right, 4)))
        if len(rows) >= 200:
            _flush(csv_path, rows)
        save_q.task_done()


# ── TCP receive helper ───────────────────────────────────────────────────────────

def _recv_exact(sock, n):
    buf = bytearray(n)
    mv  = memoryview(buf)
    pos = 0
    while pos < n:
        got = sock.recv_into(mv[pos:], n - pos)
        if not got:
            raise ConnectionError('laptop disconnected')
        pos += got
    return bytes(buf)


# ── TCP thread — updates motor state and drives ESP32 ───────────────────────────

def _serve_session(conn, motors, frame_idx_ref):
    global _recording, _left, _right
    was_recording = False

    try:
        while _running:
            data = _recv_exact(conn, _PKT.size)
            rec_byte, left, right = _PKT.unpack(data)
            recording = bool(rec_byte)

            motors.send(left, right)

            with _state_lock:
                _recording = recording
                _left      = left
                _right     = right

            if recording != was_recording:
                print(f'[rec] {"ON" if recording else "OFF"}  ({frame_idx_ref[0]} frames so far)',
                      flush=True)
                was_recording = recording

    except (ConnectionError, OSError) as e:
        print(f'[cmd] {e}')
    finally:
        with _state_lock:
            _recording = False


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    global _running

    # ── Camera ───────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        sys.exit('Could not open CSI camera — check ribbon cable and pipeline settings')

    frame_idx_ref = [0]
    # Queue sized for ~20 s of frames; blocking put prevents silent drops
    save_q = queue.Queue(maxsize=600)

    threading.Thread(target=_capture_loop, args=(cap, save_q, frame_idx_ref), daemon=True).start()

    print('Waiting for first camera frame...')
    time.sleep(1.0)   # GStreamer pipeline takes ~0.5 s to start
    print(f'[cam] ready  ({IMG_W}×{IMG_H})')

    # ── Motors ────────────────────────────────────────────────────────────────────
    motors = motor_client.connect()
    print('[motor] ESP32 connected')

    # ── Session directory ─────────────────────────────────────────────────────────
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    sess     = Path(__file__).resolve().parent / f'../data/session_{ts}'
    fdir     = sess / 'frames'
    csv_path = sess / 'labels.csv'
    fdir.mkdir(parents=True, exist_ok=True)
    print(f'[sess] saving to {sess}')

    # ── Save thread ───────────────────────────────────────────────────────────────
    threading.Thread(target=_save_loop, args=(save_q, fdir, csv_path), daemon=True).start()

    # ── Shutdown handler ──────────────────────────────────────────────────────────
    def _shutdown(sig, _frame):
        global _running
        print('\n[shutdown] stopping...')
        _running = False

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── TCP server — accepts one laptop at a time, auto-reconnects ────────────────
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('', LABEL_PORT))
    srv.listen(1)
    srv.settimeout(1.0)
    print(f'Listening for laptop on port {LABEL_PORT}')

    try:
        while _running:
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue

            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f'[cmd] laptop connected from {addr}')
            _serve_session(conn, motors, frame_idx_ref)
            motors.send(0.0, 0.0)
            conn.close()
            print(f'[cmd] laptop disconnected  ({frame_idx_ref[0]} frames saved)')

    finally:
        _running = False
        motors.send(0.0, 0.0)
        motors.close()
        save_q.put(None)
        save_q.join()
        cap.release()
        srv.close()
        print(f'[done] {frame_idx_ref[0]} frames → {sess}')


if __name__ == '__main__':
    main()
