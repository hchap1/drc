# pygame_viewer.py

import io
import queue
import socket
import struct
import threading
import time

import pygame

from client import connect

JETSON_IP = '192.168.4.1'
PORT = 5007
_HEADER = struct.Struct('<I')
RECONNECT_DELAY = 1.0   # seconds to wait between reconnect attempts


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError('server closed the connection')
        buf += chunk
    return buf


def _receiver(ip: str, port: int, frame_queue: queue.Queue,
              stop_event: threading.Event) -> None:
    """Background thread: connects, receives frames, auto-reconnects on any
    error. Never puts an exception into frame_queue -- the main loop never
    needs to know about transient video dropouts."""
    while not stop_event.is_set():
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.0)            # don't hang forever on connect
            sock.connect((ip, port))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(None)           # blocking recv from here on
            print(f'Video: connected to {ip}:{port}')

            while not stop_event.is_set():
                (length,) = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
                jpeg_bytes = _recv_exact(sock, length)

                # Always keep only the latest frame
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                frame_queue.put(jpeg_bytes)

        except Exception as exc:
            print(f'Video: {exc} -- reconnecting in {RECONNECT_DELAY}s')
        finally:
            if sock:
                try:
                    sock.close()
                except OSError:
                    pass

        # Brief pause so we don't spam-reconnect on a hard failure
        stop_event.wait(RECONNECT_DELAY)


def main(ip: str = JETSON_IP, port: int = PORT) -> None:
    c = connect(JETSON_IP)

    stop_event = threading.Event()
    frame_queue: queue.Queue = queue.Queue(maxsize=1)
    t = threading.Thread(
        target=_receiver, args=(ip, port, frame_queue, stop_event), daemon=True
    )
    t.start()

    pygame.init()
    screen = pygame.display.set_mode((640, 480), pygame.RESIZABLE)
    pygame.display.set_caption('Jetson feed')
    clock = pygame.time.Clock()
    font = pygame.font.Font('freesansbold.ttf', 25)

    speed = 0.1
    running = True

    while running:
        dt = clock.tick(100)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

        key = pygame.key.get_pressed()

        left = 0.0
        right = 0.0

        if key[pygame.K_w]:
            left += speed
            right += speed
        if key[pygame.K_d]:
            left += speed * 3
            right -= speed * 3
        if key[pygame.K_a]:
            left -= speed * 3
            right += speed * 3
        if key[pygame.K_s]:
            left -= speed
            right -= speed

        if key[pygame.K_SPACE]:
            speed += dt / 10000
        if key[pygame.K_LSHIFT]:
            speed -= dt / 10000

        speed = max(0.05, min(0.9, speed))

        c.send(left, right)

        # --- Frame display (non-blocking; keeps last frame if nothing new) ---
        try:
            jpeg_bytes = frame_queue.get_nowait()
            surf = pygame.image.load(io.BytesIO(jpeg_bytes))
            if surf.get_size() != screen.get_size():
                screen = pygame.display.set_mode(surf.get_size(), pygame.RESIZABLE)
            screen.blit(surf, (0, 0))
            screen.blit(
                font.render(f'SPEED {speed:.2f}', True, (255, 0, 0)),
                (10, 10)
            )
            pygame.display.flip()
        except queue.Empty:
            pass   # no new frame yet -- don't redraw, just continue the loop

    stop_event.set()
    c.close()
    pygame.quit()


if __name__ == '__main__':
    main()
