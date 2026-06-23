import cv2

import video_server

from process_cv import process_frame

CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
FRAMERATE = 30
FLIP_METHOD = 0


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=CAPTURE_WIDTH,
    capture_height=CAPTURE_HEIGHT,
    framerate=FRAMERATE,
    flip_method=FLIP_METHOD,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, "
        "framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
        )
    )


def main():
    cap = cv2.VideoCapture(
        gstreamer_pipeline(),
        cv2.CAP_GSTREAMER,
    )

    if not cap.isOpened():
        raise RuntimeError(
            'Could not open CSI camera -- check the ribbon cable and pipeline settings'
        )

    video = video_server.serve()

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                continue

            flipped = cv2.flip(frame, -1)

            video.send(flipped)

    finally:
        cap.release()
        video.close()


if __name__ == '__main__':
    main()
