"""
jetson_main_cnn.py — autonomous driving via the trained CNN.

Usage:
  python3 jetson_main_cnn.py <folder>           loads <folder>/model.pt
  python3 jetson_main_cnn.py <folder> 1.05      same with 5% speed boost

  python3 view_camera_feed.py    (laptop) — watch the debug feed on port 5007
"""

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np
import torch

import motor_client
import video_server
from cnn_model import DrivingCNN, preprocess, IMG_W, IMG_H
DEBUG_SKIP = 2    # stream every Nth frame on port 5007

SENSOR_W    = 1280
SENSOR_H    = 720
FRAMERATE   = 30
FLIP_METHOD = 2   # 180° — camera is upside-down


# ── GStreamer pipeline — capture directly at CNN input resolution ──────────────

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


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model(args):
    path = os.path.join(args.folder, 'model.pt')
    if not os.path.exists(path):
        sys.exit(f'Model not found: {path}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = torch.jit.load(path, map_location=device).eval()
    print(f'[model] TorchScript on {device}  ← {path}')
    return model, device


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _running

    ap = argparse.ArgumentParser()
    ap.add_argument('folder',       help='model folder containing model.pt (e.g. run1)')
    ap.add_argument('speed_mult',   nargs='?', type=float, default=1.0,
                    help='motor output multiplier, e.g. 1.05 = 5%% faster (default 1.0)')
    ap.add_argument('straight_mult', nargs='?', type=float, default=1.0,
                    help='extra multiplier applied only when L/R powers are within 5%% of each other (default 1.0)')
    args = ap.parse_args()

    model, device_str = _load_model(args)
    device = torch.device(device_str)
    speed_mult    = args.speed_mult
    straight_mult = args.straight_mult
    if speed_mult != 1.0:
        print(f'[speed] multiplier {speed_mult:.3f}x')
    if straight_mult != 1.0:
        print(f'[straight] multiplier {straight_mult:.3f}x (applied when L/R within 5%%)')

    # warm-up pass to avoid latency spike on first real frame
    with torch.no_grad():
        dummy = torch.zeros(1, 3, IMG_H, IMG_W).to(device)
        model(dummy)
    print('[model] warm-up done')

    cap = cv2.VideoCapture(_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        sys.exit('Could not open CSI camera')

    threading.Thread(target=_capture_loop, args=(cap,), daemon=True).start()

    motors    = motor_client.connect()
    # stream_width=None → send at native 160×90 so viewer sees exactly what the CNN sees
    debug_vid = video_server.serve(port=5007, jpeg_quality=80, stream_width=None)
    print(f'Debug feed on port 5007  ({IMG_W}×{IMG_H})')

    # Wait for the camera pipeline to produce its first frame before prompting,
    # so that pressing Enter starts driving immediately with no delay.
    print('Waiting for camera...')
    while True:
        with _frame_lock:
            if _latest_frame is not None:
                break
        time.sleep(0.01)
    print('Camera ready.')

    input('Press Enter to start driving...')

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

            # Inference — frame is already IMG_W×IMG_H from GStreamer
            tensor = preprocess(frame).to(device)
            with torch.no_grad():
                out = model(tensor)
            left  = float(out[0, 0]) * speed_mult
            right = float(out[0, 1]) * speed_mult
            if abs(left - right) <= 0.08 * max(abs(left), abs(right)):
                left  *= straight_mult
                right *= straight_mult

            motors.send(left, right)
            frame_n += 1

            if frame_n % DEBUG_SKIP == 0:
                dbg = frame.copy()
                cv2.putText(dbg, f'PT  L:{left:+.3f}  R:{right:+.3f}',
                            (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                debug_vid.send(dbg)

            now = time.monotonic()
            if now - t_log >= 5.0:
                print(f'fps={frame_n/(now-t0):.1f}  L={left:+.3f}  R={right:+.3f}')
                t_log = now

    finally:
        _running = False
        cap.release()
        motors.send(0.0, 0.0)
        motors.close()
        debug_vid.close()


if __name__ == '__main__':
    main()
