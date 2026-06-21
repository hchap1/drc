# main_no_video.py
# Jetson Nano main loop WITHOUT video streaming. Camera capture + CV
# processing happen locally only; nothing goes over the network except
# motor commands (currently hardcoded to 0,0 until process_frame is wired in).

import cv2
import numpy as np

import client as motor_client

ESP32_IP = '192.168.4.1'

CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
FRAMERATE = 30
FLIP_METHOD = 0   # 0 = none. Use 2 if your camera is mounted upside down.


def gstreamer_pipeline(sensor_id=0, capture_width=CAPTURE_WIDTH, capture_height=CAPTURE_HEIGHT,
                        framerate=FRAMERATE, flip_method=FLIP_METHOD):
    """GStreamer pipeline for the CSI ribbon-cable camera via nvarguscamerasrc.
    appsink drop=true / max-buffers=1 means we always grab the freshest
    frame instead of draining a backlog -- same low-latency philosophy
    as the motor link."""
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, "
        "framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
        % (sensor_id, capture_width, capture_height, framerate, flip_method)
    )

def main():
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError('Could not open CSI camera -- check the ribbon cable and pipeline settings')

    motors = motor_client.connect(ESP32_IP)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            left, right, output_image = process_frame(frame)

            # Hardcoded to (0, 0) for now, as requested. Once you trust
            # process_frame()'s output, swap this line for:
            #     motors.send(left, right)
            motors.send(0.0, 0.0)

    finally:
        cap.release()
        motors.close()


if __name__ == '__main__':
    main()
