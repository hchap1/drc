"""
collect_data.py — run on laptop while manually driving the robot.
Sends motor commands to the ESP32 (UDP) AND streams labels to the Jetson (TCP)
so the Jetson saves frames+labels locally. No video stream needed.

Jetson should be running:  python3 jetson_collect.py

Controls
  W/A/S/D        drive
  SPACE          speed up   |   LSHIFT  speed down
  R              toggle recording on / off
  Q / ESC        quit

After the session, copy data off the Jetson:
  scp -r user@192.168.4.2:~/drc/data ./data
"""

import socket
import struct
import threading
import time

import pygame

from client import connect

JETSON_IP    = '192.168.4.2'
LABEL_PORT   = 5009          # Jetson listens here for label packets
BASE_SPEED   = 0.15
MAX_SPEED    = 0.20

# 9-byte label packet: recording(uint8) left(float32) right(float32)
_PKT = struct.Struct('<Bff')


def _label_thread(ip, port, getter, stop):
    """Continuously sends label packets to the Jetson. Auto-reconnects."""
    while not stop.is_set():
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.0)
            sock.connect((ip, port))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(None)
            print(f'[label] connected to Jetson {ip}:{port}')
            while not stop.is_set():
                recording, left, right = getter()
                sock.sendall(_PKT.pack(int(recording), left, right))
                time.sleep(0.05)   # 20 Hz
        except Exception as e:
            print(f'[label] {e} — reconnecting in 1 s')
        finally:
            if sock:
                try: sock.close()
                except OSError: pass
        stop.wait(1.0)


def main():
    motors = connect()

    # shared state read by label thread
    _state = {'recording': False, 'left': 0.0, 'right': 0.0}
    _lock  = threading.Lock()

    def getter():
        with _lock:
            return _state['recording'], _state['left'], _state['right']

    stop = threading.Event()
    threading.Thread(
        target=_label_thread, args=(JETSON_IP, LABEL_PORT, getter, stop), daemon=True
    ).start()

    pygame.init()
    screen = pygame.display.set_mode((480, 160), pygame.RESIZABLE)
    pygame.display.set_caption('DRC — Data Collection (Jetson saving)')
    font = pygame.font.SysFont(None, 28)
    big  = pygame.font.SysFont(None, 64)
    tick = pygame.time.Clock()

    recording   = False
    frame_count = 0       # approximate — actual count lives on Jetson
    speed       = BASE_SPEED
    motor_timer = 0

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
                        print(f'[rec] {tag}')

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

            motor_timer += dt
            if motor_timer >= 50:   # 20 Hz
                motors.send(left, right)
                motor_timer = 0

            with _lock:
                _state['recording'] = recording
                _state['left']      = round(left,  4)
                _state['right']     = round(right, 4)

            if recording:
                frame_count += 1   # rough estimate for HUD

            # ── display ──────────────────────────────────────────────────────
            sw, sh = screen.get_size()
            screen.fill((20, 20, 20))

            hud = f'L:{left:+.2f}  R:{right:+.2f}  spd:{speed:.2f}  ~frames:{frame_count}'
            screen.blit(font.render(hud, True, (220, 220, 220)), (8, 8))

            if recording:
                screen.blit(big.render('● REC', True, (255, 40, 40)), (8, sh - 70))
            else:
                hint = 'R=record  SPACE=faster  SHIFT=slower  Q=quit'
                screen.blit(font.render(hint, True, (130, 130, 130)), (8, sh - 30))

            pygame.display.flip()

    finally:
        stop.set()
        motors.send(0.0, 0.0)
        motors.close()
        pygame.quit()
        print('\nDone. Copy data from Jetson with:')
        print(f'  scp -r user@{JETSON_IP}:~/drc/data ./data')


if __name__ == '__main__':
    main()
