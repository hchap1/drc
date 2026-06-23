"""
jetson_collect.py — run on Jetson during manual data collection sessions.
Captures raw camera frames and streams them on port 5008 at full framerate.
No CV processing — faster than jetson_main_video.py for data collection.

On laptop: run collect_data.py
"""

import signal
import sys
import time
import cv2

import video_server
from process_cv import PROC_W, PROC_H

SENSOR_W    = 1280
SENSOR_H    = 720
FRAMERATE   = 30
FLIP_METHOD = 2   # 180° — camera is upside-down


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


def main():
    cap = cv2.VideoCapture(_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        sys.exit('Could not open CSI camera — check ribbon cable and pipeline')

    # stream_width=None sends frames at their native resolution (no resize)
    stream = video_server.serve(port=5008, jpeg_quality=80, stream_width=None)
    print(f'Streaming raw {PROC_W}×{PROC_H} frames on port 5008  (Ctrl-C to stop)')

    n = 0
    t0 = tlog = time.monotonic()

    def _shutdown(sig, frame):
        print(f'\nStopping after {n} frames')
        cap.release()
        stream.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        stream.send(frame)
        n += 1
        now = time.monotonic()
        if now - tlog >= 5.0:
            print(f'{n / (now - t0):.1f} fps')
            tlog = now


if __name__ == '__main__':
    main()
