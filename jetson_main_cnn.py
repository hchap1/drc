"""
jetson_main_cnn.py — autonomous driving via the trained CNN.

Automatically uses TensorRT if model_trt.pt exists, otherwise falls back
to the plain TorchScript model.pt.  Debug feed on port 5007 shows the
frame at exactly 160×90 — the resolution the CNN was trained on.

Workflow:
  1. python3 convert_trt.py          # one-time conversion (needs torch2trt)
  2. python3 jetson_main_cnn.py      # drives autonomously
  3. python3 view_camera_feed.py     # watch from laptop (port 5007)

Args:
  --model    override model file (default: auto-detect model_trt.pt → model.pt)
  --trt      force TensorRT even if auto-detect would pick otherwise
  --no-trt   force plain TorchScript
"""

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np
import torch

import client as motor_client
import video_server
from cnn_model import DrivingCNN, preprocess, IMG_W, IMG_H

ESP32_IP   = '192.168.4.1'
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
    use_trt = False

    path = args.model or 'model.pt'
    if not os.path.exists(path):
        sys.exit(f'Model not found: {path}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = torch.jit.load(path, map_location=device).eval()
    print(f'[model] TorchScript on {device}  ← {path}')
    return model, device, False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _running

    ap = argparse.ArgumentParser()
    ap.add_argument('--model',  default=None)
    ap.add_argument('--trt',    action='store_true')
    ap.add_argument('--no-trt', action='store_true', dest='no_trt')
    args = ap.parse_args()

    model, device_str, use_trt = _load_model(args)
    device = torch.device(device_str)

    # warm-up pass to avoid latency spike on first real frame
    with torch.no_grad():
        dummy = torch.zeros(1, 3, IMG_H, IMG_W).to(device)
        model(dummy)
    print('[model] warm-up done')

    cap = cv2.VideoCapture(_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        sys.exit('Could not open CSI camera')

    threading.Thread(target=_capture_loop, args=(cap,), daemon=True).start()

    motors    = motor_client.connect(ESP32_IP)
    # stream_width=None → send at native 160×90 so viewer sees exactly what the CNN sees
    debug_vid = video_server.serve(port=5007, jpeg_quality=80, stream_width=None)
    print(f'Debug feed on port 5007  ({IMG_W}×{IMG_H})')

    # ── Wait for viewer then Enter ────────────────────────────────────────────
    print('Waiting for view_camera_feed.py to connect...')
    while not debug_vid._clients:
        debug_vid._accept_new_clients()
        with _frame_lock:
            frame = _latest_frame
        if frame is not None:
            debug_vid.send(frame)
        time.sleep(0.05)

    print(f'{len(debug_vid._clients)} viewer(s) connected.')
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
            left  = float(out[0, 0])
            right = float(out[0, 1])

            motors.send(left, right)
            frame_n += 1

            if frame_n % DEBUG_SKIP == 0:
                dbg = frame.copy()
                mode = 'TRT' if use_trt else 'PT'
                cv2.putText(dbg, f'{mode}  L:{left:+.3f}  R:{right:+.3f}',
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
