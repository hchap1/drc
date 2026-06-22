# main_video.py
# Same as main_no_video.py but also broadcasts output_image over UDP so
# video_client.py can display it elsewhere on the network.

import cv2
import numpy as np

import client as motor_client
import video_server

from process_cv import process_frame

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
    video = video_server.serve()       # listens on 0.0.0.0:5007 for TCP clients

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue


            flipped = cv2.flip(frame, -1)
            left, right, output_image = process_frame(flipped)

            # Hardcoded to (0, 0) for now, as requested. Once you trust
            # process_frame()'s output, swap this line for:
            #     motors.send(left, right)
            motors.send(left, right)
            video.send(output_image)

    finally:
        cap.release()
        motors.close()
        video.close()


if __name__ == '__main__':
    main()
