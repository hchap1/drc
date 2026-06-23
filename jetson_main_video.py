# jetson_main_video.py
# Runs the line-following controller and broadcasts two video streams:
#   port 5007 - debug overlay (for viewing)
#   port 5008 - raw frames   (for hsv_tune.py)

import threading
import time
import cv2

import client as motor_client
import video_server
from process_cv import process_frame, PROC_W, PROC_H

ESP32_IP = '192.168.4.1'

SENSOR_WIDTH  = 1280
SENSOR_HEIGHT = 720
FRAMERATE     = 30

# 2 = 180° rotation (camera mounted upside-down). Use 0 if right-side up.
FLIP_METHOD   = 2

DEBUG_SKIP = 5   # send debug frame every Nth frame
RAW_SKIP   = 1   # send raw frame every Nth frame (1=every frame for collect_data.py)


def gstreamer_pipeline(sensor_id=0):
    """
    Captures from CSI sensor, flips + downscales to PROC_W x PROC_H entirely
    in NVMM (Jetson video engine) before handing pixels to the CPU.
    """
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=%d, height=%d, framerate=%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw(memory:NVMM), width=%d, height=%d ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
        % (sensor_id,
           SENSOR_WIDTH, SENSOR_HEIGHT, FRAMERATE,
           FLIP_METHOD,
           PROC_W, PROC_H)
    )


# ── Capture thread ────────────────────────────────────────────────────────────
# Decouples GStreamer from CV processing so we always consume the latest frame
# without letting cap.read() block the control loop.

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


def main():
    global _running

    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(
            'Could not open CSI camera -- check the ribbon cable and pipeline settings'
        )

    threading.Thread(target=_capture_loop, args=(cap,), daemon=True).start()

    motors    = motor_client.connect(ESP32_IP)
    debug_vid = video_server.serve(port=5007)
    raw_vid   = video_server.serve(port=5008, stream_width=None)   # full-res for collect_data.py

    frame_n = 0
    t0      = time.monotonic()
    t_log   = t0

    try:
        while True:
            with _frame_lock:
                frame = _latest_frame

            if frame is None:
                continue

            left, right, debug = process_frame(frame)
            motors.send(left, right)

            frame_n += 1

            if frame_n % DEBUG_SKIP == 0:
                debug_vid.send(debug)
            if frame_n % RAW_SKIP == 0:
                raw_vid.send(frame)

            now = time.monotonic()
            if now - t_log >= 5.0:
                fps = frame_n / (now - t0)
                print(f'fps={fps:.1f}  L={left:+.3f}  R={right:+.3f}')
                t_log = now

    finally:
        _running = False
        cap.release()
        motors.close()
        debug_vid.close()
        raw_vid.close()


if __name__ == '__main__':
    main()
