"""
jetson_main_cnn.py — autonomous driving via the trained CNN.
Drop model.pt (output of train_cnn.py) into the same directory, then run:

  python3 jetson_main_cnn.py
  python3 jetson_main_cnn.py --model /path/to/model.pt

Streams a debug view (motor values overlaid) on port 5007 so you can
watch from a laptop using  python3 view_camera_feed.py.
"""

import argparse
import sys
import threading
import time

import cv2
import numpy as np
import torch

import client as motor_client
import video_server
from process_cv import PROC_W, PROC_H

ESP32_IP  = '192.168.4.1'
IMG_W     = 160
IMG_H     = 90
DEBUG_SKIP = 3

SENSOR_W    = 1280
SENSOR_H    = 720
FRAMERATE   = 30
FLIP_METHOD = 2


# ── Preprocessing — MUST match train_cnn.py exactly ──────────────────────────

def preprocess(bgr: np.ndarray) -> torch.Tensor:
    """BGR numpy (H,W,3) → float tensor (1,3,H,W) in [-1, 1]."""
    rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    t     = torch.from_numpy(small).float().permute(2, 0, 1) / 128.0 - 1.0
    return t.unsqueeze(0)   # add batch dim


# ── GStreamer pipeline ────────────────────────────────────────────────────────

def _pipeline():
    return (
        f"nvarguscamerasrc sensor-id=0 ! "
        f"video/x-raw(memory:NVMM), width={SENSOR_W}, height={SENSOR_H}, framerate={FRAMERATE}/1 ! "
        f"nvvidconv flip-method={FLIP_METHOD} ! "
        f"video/x-raw(memory:NVMM), width={PROC_W}, height={PROC_H} ! "
        f"nvvidconv ! "
        f"video/x-raw, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


# ── Capture daemon ────────────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _running

    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='model.pt', help='TorchScript model file')
    args = ap.parse_args()

    # Load model
    try:
        model = torch.jit.load(args.model, map_location='cpu')
    except FileNotFoundError:
        sys.exit(f'Model not found: {args.model}\nTrain with:  python train_cnn.py')
    model.eval()
    print(f'Loaded model: {args.model}')

    # Warm up inference (avoids first-frame latency spike)
    with torch.no_grad():
        model(torch.zeros(1, 3, IMG_H, IMG_W))

    cap = cv2.VideoCapture(_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        sys.exit('Could not open CSI camera')

    threading.Thread(target=_capture_loop, args=(cap,), daemon=True).start()

    motors    = motor_client.connect(ESP32_IP)
    debug_vid = video_server.serve(port=5007)

    frame_n = 0
    left    = 0.0
    right   = 0.0
    t0      = time.monotonic()
    t_log   = t0

    try:
        while True:
            with _frame_lock:
                frame = _latest_frame

            if frame is None:
                continue

            # ── CNN inference ─────────────────────────────────────────────────
            tensor = preprocess(frame)
            with torch.no_grad():
                out   = model(tensor)
            left  = float(out[0, 0])
            right = float(out[0, 1])

            motors.send(left, right)
            frame_n += 1

            # ── Debug video ───────────────────────────────────────────────────
            if frame_n % DEBUG_SKIP == 0:
                dbg = frame.copy()
                cv2.putText(
                    dbg, f'CNN  L:{left:+.3f}  R:{right:+.3f}',
                    (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
                )
                debug_vid.send(dbg)

            now = time.monotonic()
            if now - t_log >= 5.0:
                fps = frame_n / (now - t0)
                print(f'fps={fps:.1f}  L={left:+.3f}  R={right:+.3f}')
                t_log = now

    finally:
        _running = False
        cap.release()
        motors.send(0.0, 0.0)
        motors.close()
        debug_vid.close()


if __name__ == '__main__':
    main()
