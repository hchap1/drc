"""
collect_data.py — run on laptop while manually driving the robot.
Sends motor commands to the Jetson (TCP, port 5009). The Jetson forwards
them to the ESP32 and saves frames + labels locally — no video stream needed.

Jetson should be running:  python3 jetson_collect.py

Controls
  W/A/S/D        drive  (additive — W+A combines)
  SPACE          speed up   |   LSHIFT  speed down
  R              toggle recording on / off
  Q / ESC        quit

After the session, copy data from the Jetson:
  scp -r user@192.168.4.1:~/drc/data ./data
"""

import socket
import struct
import threading
import time

import pygame

JETSON_IP    = '192.168.4.1'
LABEL_PORT   = 5009
SPEED        = 0.24       # starting speed (SPACE/LSHIFT adjust)
CORNER_SPEED = 0.20       # forward power is capped to this when turning
STEER_OUTER  = 1.3        # outside wheel boost during turn
STEER_INNER  = -1.2       # inside wheel — negative overrides W so it reverses even mid-drive
RAMP_UP      = 0.05       # seconds: 0 → full (press)
RAMP_DOWN    = 0.05       # seconds: full → 0 (release) — slight linger for tap corrections

_PKT = struct.Struct('<Bff')   # recording(uint8), left(float32), right(float32)


def _cmd_thread(ip, port, getter, stop):
    """Send motor+recording packets to Jetson. Auto-reconnects."""
    while not stop.is_set():
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.0)
            sock.connect((ip, port))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(None)
            print(f'[cmd] connected to Jetson {ip}:{port}')
            while not stop.is_set():
                recording, left, right = getter()
                sock.sendall(_PKT.pack(int(recording), left, right))
                time.sleep(0.033)   # 30 Hz
        except Exception as e:
            print(f'[cmd] {e} — reconnecting in 1 s')
        finally:
            if sock:
                try: sock.close()
                except OSError: pass
        stop.wait(1.0)


def _ramp(current, target, step_up, step_down):
    """Move current toward target, using different rates for up vs down."""
    diff = target - current
    if diff > 0:
        return current + min(diff, step_up)
    elif diff < 0:
        return current + max(diff, -step_down)
    return current


def main():
    _state = {'recording': False, 'left': 0.0, 'right': 0.0}
    _lock  = threading.Lock()

    def getter():
        with _lock:
            return _state['recording'], _state['left'], _state['right']

    stop = threading.Event()
    threading.Thread(target=_cmd_thread, args=(JETSON_IP, LABEL_PORT, getter, stop), daemon=True).start()

    pygame.init()
    screen = pygame.display.set_mode((480, 160), pygame.RESIZABLE)
    pygame.display.set_caption('DRC — Data Collection')
    font = pygame.font.SysFont(None, 28)
    big  = pygame.font.SysFont(None, 64)
    tick = pygame.time.Clock()

    recording = False
    speed     = SPEED
    smooth_l  = 0.0
    smooth_r  = 0.0

    try:
        while True:
            dt = tick.tick(60)   # ms

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return
                if ev.type == pygame.KEYDOWN:
                    if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                        return
                    if ev.key == pygame.K_r:
                        recording = not recording
                        print(f'[rec] {"ON" if recording else "OFF"}')

            keys  = pygame.key.get_pressed()
            target_l = target_r = 0.0

            turning = keys[pygame.K_a] or keys[pygame.K_d]
            fwd = min(speed, CORNER_SPEED) if turning else speed

            if keys[pygame.K_w]:
                target_l += fwd
                target_r += fwd
            if keys[pygame.K_s]:
                target_l -= fwd
                target_r -= fwd
            if keys[pygame.K_a]:
                target_l += speed * STEER_INNER   # inside wheel (negative = reverse)
                target_r += speed * STEER_OUTER   # outside wheel
            if keys[pygame.K_d]:
                target_l += speed * STEER_OUTER   # outside wheel
                target_r += speed * STEER_INNER   # inside wheel (negative = reverse)

            if keys[pygame.K_SPACE]:  speed += dt / 10000
            if keys[pygame.K_LSHIFT]: speed -= dt / 10000
            speed = max(0.05, min(0.9, speed))

            # clamp targets
            max_out   = speed * STEER_OUTER
            target_l  = max(-max_out, min(max_out, target_l))
            target_r  = max(-max_out, min(max_out, target_r))

            # smooth: asymmetric ramp — slow up, fast down
            max_out    = speed * STEER_OUTER
            step_up    = (max_out / RAMP_UP)   * (dt / 1000)
            step_down  = (max_out / RAMP_DOWN) * (dt / 1000)
            smooth_l   = _ramp(smooth_l, target_l, step_up, step_down)
            smooth_r   = _ramp(smooth_r, target_r, step_up, step_down)

            with _lock:
                _state['recording'] = recording
                _state['left']      = round(smooth_l, 4)
                _state['right']     = round(smooth_r, 4)

            # ── display ──────────────────────────────────────────────────────
            sw, sh = screen.get_size()
            screen.fill((20, 20, 20))
            screen.blit(font.render(
                f'L:{smooth_l:+.2f}  R:{smooth_r:+.2f}  spd:{speed:.2f}',
                True, (220, 220, 220)), (8, 8))

            if recording:
                screen.blit(big.render('● REC', True, (255, 40, 40)), (8, sh - 70))
            else:
                screen.blit(font.render('R=record  WASD=drive  SPACE/SHIFT=speed  Q=quit',
                                        True, (130, 130, 130)), (8, sh - 30))

            pygame.display.flip()

    finally:
        stop.set()
        pygame.quit()
        print('\nDone. Copy data from Jetson:')
        print(f'  scp -r user@{JETSON_IP}:~/drc/data ./data')


if __name__ == '__main__':
    main()
