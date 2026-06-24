"""
jetson_collect.py — run on Jetson during manual data collection.
Receives motor commands from the laptop (port 5009), forwards them to the
ESP32, and saves frames + labels locally when recording is active.

No video stream, no label channel — laptop just sends drive commands here.

Usage:
  Jetson:  python3 jetson_collect.py
  Laptop:  python3 collect_data.py

After the session, copy data to laptop/PC for training:
  scp -r user@192.168.4.2:~/drc/data ./data
"""

import csv
import signal
import socket
import struct
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

import client as motor_client
from cnn_model import IMG_W, IMG_H

LABEL_PORT  = 5009
ESP32_IP    = '192.168.4.1'
JPEG_Q      = 90

SENSOR_W    = 1280
SENSOR_H    = 720
FRAMERATE   = 30
FLIP_METHOD = 2

_PKT = struct.Struct('<Bff')   # recording(uint8), left(float32), right(float32)


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


_latest_frame = None
_frame_lock   = threading.Lock()
_running      = True


def _capture_loop(cap):
    global _latest_frame
    while _running:
        ok, frame = cap.read()
        if ok:
            with _frame_lock:
                _latest_frame = frame


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


def _serve_session(conn, sess_dir, motors):
    fdir     = sess_dir / 'frames'
    csv_path = sess_dir / 'labels.csv'
    fdir.mkdir(parents=True, exist_ok=True)

    rows      = []
    frame_idx = 0
    was_rec   = False

    print(f'[session] saving to {sess_dir}')
    try:
        while True:
            data = _recv_exact(conn, _PKT.size)
            recording, left, right = _PKT.unpack(data)
            recording = bool(recording)

            # forward to ESP32
            motors.send(left, right)

            if recording and not was_rec:
                print('[rec] ON')
            elif not recording and was_rec:
                print(f'[rec] OFF — {frame_idx} frames saved')
            was_rec = recording

            if recording:
                with _frame_lock:
                    frame = _latest_frame
                if frame is None:
                    continue

                fname = f'{frame_idx:06d}.jpg'
                ok, enc = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
                if ok:
                    (fdir / fname).write_bytes(enc.tobytes())
                    rows.append((fname, round(left, 4), round(right, 4)))
                    frame_idx += 1
                    if frame_idx % 200 == 0:
                        _flush(csv_path, rows)
                        print(f'[rec] {frame_idx} frames saved')

    except Exception as e:
        print(f'[session] ended: {e}')
    finally:
        motors.send(0.0, 0.0)
        _flush(csv_path, rows)
        print(f'[session] {frame_idx} frames total → {sess_dir}')


def main():
    global _running

    cap = cv2.VideoCapture(_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        sys.exit('Could not open CSI camera')

    threading.Thread(target=_capture_loop, args=(cap,), daemon=True).start()

    print('Waiting for camera...')
    while True:
        with _frame_lock:
            if _latest_frame is not None:
                break
        time.sleep(0.05)
    print(f'Camera ready  ({IMG_W}×{IMG_H})')

    motors = motor_client.connect(ESP32_IP)
    print(f'ESP32 connected at {ESP32_IP}')

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('', LABEL_PORT))
    srv.listen(1)
    print(f'Listening for laptop on port {LABEL_PORT}  (Ctrl-C to stop)')

    def _shutdown(sig, frame):
        global _running
        _running = False
        motors.send(0.0, 0.0)
        motors.close()
        cap.release()
        srv.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)

    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f'[label] laptop connected from {addr}')
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        sess = Path(f'data/session_{ts}')
        _serve_session(conn, sess, motors)
        conn.close()


if __name__ == '__main__':
    main()
