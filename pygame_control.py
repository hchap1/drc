# pygame_viewer.py
# Pygame-based viewer for the Jetson camera feed served by video_server.py.
# Networking is identical to video_client.py (TCP, 4-byte length-prefixed
# JPEG frames). The receive loop runs in a background thread so the pygame
# event loop (and therefore the window) stays responsive at all times.
# The window auto-resizes to match the first frame and any subsequent size
# changes. Press Q or Escape or close the window to quit.

import io
import queue
import socket
import struct
import threading
import pygame

from client import connect

JETSON_IP = '192.168.4.2'
PORT = 5007
_HEADER = struct.Struct('<I')

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError('server closed the connection')
        buf += chunk
    return buf


def _receiver(sock: socket.socket, frame_queue: queue.Queue) -> None:
    """Background thread: receives JPEG frames and pushes them to the queue.
    Keeps only the latest frame -- drops any unread one before pushing --
    so the display always shows the freshest image and never falls behind."""
    try:
        while True:
            (length,) = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
            jpeg_bytes = _recv_exact(sock, length)

            # Evict stale frame before pushing the new one
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            frame_queue.put(jpeg_bytes)

    except Exception as exc:
        # Signal the main thread that the connection is gone
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
        frame_queue.put(exc)


def main(ip: str = JETSON_IP, port: int = PORT) -> None:
    c = connect()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f'Connected to video server at {ip}:{port}')

    frame_queue: queue.Queue = queue.Queue(maxsize=1)
    t = threading.Thread(target=_receiver, args=(sock, frame_queue), daemon=True)
    t.start()

    pygame.init()
    # Start with a placeholder window; it resizes to the actual frame on
    # the first received image
    screen = pygame.display.set_mode((640, 480), pygame.RESIZABLE)
    pygame.display.set_caption('Jetson feed')
    clock = pygame.time.Clock()

    left = 0
    right = 0

    speed = 100

    running = True
    while running:

        dt = clock.tick(100)
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

        key = pygame.key.get_pressed()
        if key[pygame.K_w]:
            left += speed
            right += speed
        if key[pygame.K_d]:
            left += speed
            right -= speed
        if key[pygame.K_a]:
            left -= speed
            right += speed
        if key[pygame.K_s]:
            left -= speed
            right -= speed

        if key[pygame.K_SPACE]:
            speed += dt / 10
        if key[pygame.K_LSHIFT]:
            speed -= dt / 10

        if speed < 50: speed = 50
        if speed > 500: speed = 500

        # --- Frame display ---
        try:
            item = frame_queue.get_nowait()

            if isinstance(item, Exception):
                print(f'Connection lost: {item}')
                running = False
            else:
                # Decode JPEG via pygame (no OpenCV needed for display)
                surf = pygame.image.load(io.BytesIO(item))

                # Auto-resize the window on first frame or if frame dimensions change
                if surf.get_size() != screen.get_size():
                    screen = pygame.display.set_mode(surf.get_size(), pygame.RESIZABLE)

                screen.blit(surf, (0, 0))
                pygame.display.flip()

        except queue.Empty:
            pass

        c.send(left, right)

    sock.close()
    pygame.quit()


if __name__ == '__main__':
    main()
