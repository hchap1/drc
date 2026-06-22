# jetson_main_video.py
# Runs the line-following controller and broadcasts a debug video feed over TCP.
# Use video_server.py (client side) to view the feed.

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

VIDEO_SKIP = 5   # encode + send debug frame only every Nth control frame


def gstreamer_pipeline(sensor_id=0):
    """
    Captures from CSI sensor, flips in NVMM, then hardware-downscales to
    PROC_W x PROC_H before handing pixels to the CPU.  All heavy lifting
    (flip + scale) stays on the Jetson's video engine.
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


def main():
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(
            'Could not open CSI camera -- check the ribbon cable and pipeline settings'
        )

    motors = motor_client.connect(ESP32_IP)
    video  = video_server.serve()

    frame_n = 0
    t0      = time.monotonic()
    t_log   = t0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            left, right, debug = process_frame(frame)
            motors.send(left, right)

            frame_n += 1

            if frame_n % VIDEO_SKIP == 0:
                video.send(debug)

            now = time.monotonic()
            if now - t_log >= 5.0:
                fps = frame_n / (now - t0)
                print(f'fps={fps:.1f}  L={left:+.3f}  R={right:+.3f}')
                t_log = now

    finally:
        cap.release()
        motors.close()
        video.close()


if __name__ == '__main__':
    main()
