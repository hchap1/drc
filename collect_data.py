"""
collect_data.py — run on laptop while manually driving the robot.
Streams raw camera frames from the Jetson (port 5008) and saves each frame
+ motor label to ./data/session_TIMESTAMP/.

Jetson should be running:  python3 jetson_collect.py   (fast, no CV overhead)

Controls
  W/A/S/D        drive  (same as pygame_control.py)
  SPACE          speed up   |   LSHIFT  speed down
  R              toggle recording on / off
  Q / ESC        quit  (auto-saves CSV)

Output layout
  data/
    session_20241015_143022/
      frames/
        000000.jpg  000001.jpg  ...   (160×90 JPEG)
      labels.csv                       frame,left,right
"""

import csv
import io
import queue
import socket
import struct
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pygame

from client import connect

JETSON_IP  = '192.168.4.2'
VIDEO_PORT = 5008
SAVE_W     = 80
SAVE_H     = 45
JPEG_Q     = 90       # quality for saved training frames
BASE_SPEED = 0.15
MAX_SPEED  = 0.20

_HEADER = struct.Struct('<I')
_SAVE_Q: queue.Queue = queue.Queue(maxsize=120)   # ~4 s buffer at 30fps


# ── Reliable receive helper ───────────────────────────────────────────────────

def _recv_exact(sock, n):
    buf = bytearray(n)
    mv  = memoryview(buf)
    pos = 0
    while pos < n:
        got = sock.recv_into(mv[pos:], n - pos)
        if not got:
            raise ConnectionError('server disconnected')
        pos += got
    return bytes(buf)


# ── Background video receiver (auto-reconnects) ───────────────────────────────

def _video_thread(ip, port, fq, stop):
    while not stop.is_set():
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.0)
            sock.connect((ip, port))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(None)
            print(f'[video] connected to {ip}:{port}')
            while not stop.is_set():
                (n,) = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
                data = _recv_exact(sock, n)
                try:    fq.get_nowait()
                except queue.Empty: pass
                fq.put(data)
        except Exception as e:
            print(f'[video] {e} — reconnecting in 1 s')
        finally:
            if sock:
                try: sock.close()
                except OSError: pass
        stop.wait(1.0)


# ── CSV flusher (append so partial sessions survive crashes) ──────────────────

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


# ── Background save worker ────────────────────────────────────────────────────

def _save_worker(fdir: Path, csv_path: Path, stop: threading.Event):
    rows: list = []
    while not stop.is_set() or not _SAVE_Q.empty():
        try:
            fname, left, right, jpg_bytes = _SAVE_Q.get(timeout=0.1)
        except queue.Empty:
            continue
        arr = np.frombuffer(jpg_bytes, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            small = cv2.resize(bgr, (SAVE_W, SAVE_H), interpolation=cv2.INTER_AREA)
            ok, enc = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
            if ok:
                (fdir / fname).write_bytes(enc.tobytes())
        rows.append((fname, left, right))
        if len(rows) >= 200:
            _flush(csv_path, rows)
    _flush(csv_path, rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    sess = Path(f'data/session_{ts}')
    fdir = sess / 'frames'
    fdir.mkdir(parents=True, exist_ok=True)
    csv_path = sess / 'labels.csv'
    print(f'Session directory: {sess}')

    motors = connect()

    stop     = threading.Event()
    save_stop = threading.Event()
    fq       = queue.Queue(maxsize=1)
    threading.Thread(
        target=_video_thread, args=(JETSON_IP, VIDEO_PORT, fq, stop), daemon=True
    ).start()
    threading.Thread(
        target=_save_worker, args=(fdir, csv_path, save_stop), daemon=True
    ).start()

    pygame.init()
    screen = pygame.display.set_mode((640, 360), pygame.RESIZABLE)
    pygame.display.set_caption('DRC — Data Collection')
    font  = pygame.font.SysFont(None, 28)
    big   = pygame.font.SysFont(None, 64)
    tick  = pygame.time.Clock()

    recording  = False
    frame_idx  = 0
    speed      = BASE_SPEED
    latest_jpg = None   # raw JPEG bytes, kept for display

    try:
        while True:
            dt = tick.tick(60)

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return
                if ev.type == pygame.KEYDOWN:
                    if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                        return
                    if ev.key == pygame.K_r:
                        recording = not recording
                        tag = 'ON ' if recording else 'OFF'
                        print(f'[rec] {tag} — {frame_idx} frames saved')

            keys  = pygame.key.get_pressed()
            left  = right = 0.0

            if keys[pygame.K_w]: left  += speed;       right += speed
            if keys[pygame.K_s]: left  -= speed;       right -= speed
            if keys[pygame.K_a]: left  -= speed * 2.5; right += speed * 2.5
            if keys[pygame.K_d]: left  += speed * 2.5; right -= speed * 2.5

            if keys[pygame.K_SPACE]:  speed += dt / 10000
            if keys[pygame.K_LSHIFT]: speed -= dt / 10000
            speed = max(0.05, min(MAX_SPEED, speed))

            left  = max(-MAX_SPEED, min(MAX_SPEED, left))
            right = max(-MAX_SPEED, min(MAX_SPEED, right))
            motors.send(left, right)

            # pull latest frame
            try:
                latest_jpg = fq.get_nowait()
            except queue.Empty:
                pass

            # save if recording — push to background thread, never block main loop
            if latest_jpg is not None and recording:
                fname = f'{frame_idx:06d}.jpg'
                try:
                    _SAVE_Q.put_nowait((fname, round(left, 4), round(right, 4), latest_jpg))
                    frame_idx += 1
                    if frame_idx % 200 == 0:
                        print(f'[rec] {frame_idx} frames queued')
                except queue.Full:
                    print('[rec] save queue full — frame dropped')

            # ── display ──────────────────────────────────────────────────────
            sw, sh = screen.get_size()
            if latest_jpg is not None:
                try:
                    surf = pygame.image.load(io.BytesIO(latest_jpg))
                    surf = pygame.transform.scale(surf, (sw, sh))
                    screen.blit(surf, (0, 0))
                except Exception:
                    screen.fill((30, 30, 30))
            else:
                screen.fill((30, 30, 30))
                screen.blit(font.render('Waiting for video…', True, (160, 160, 160)), (sw // 2 - 80, sh // 2))

            hud = f'L:{left:+.2f}  R:{right:+.2f}  spd:{speed:.2f}  saved:{frame_idx}'
            screen.blit(font.render(hud, True, (220, 220, 220)), (8, 8))

            if recording:
                screen.blit(big.render('● REC', True, (255, 40, 40)), (8, sh - 70))
            else:
                hint = 'R=record  SPACE=faster  SHIFT=slower  Q=quit'
                screen.blit(font.render(hint, True, (130, 130, 130)), (8, sh - 30))

            pygame.display.flip()

    finally:
        stop.set()
        save_stop.set()
        motors.send(0.0, 0.0)
        motors.close()
        pygame.quit()
        print(f'\nDone. {frame_idx} frames saved to {sess}')


if __name__ == '__main__':
    main()
