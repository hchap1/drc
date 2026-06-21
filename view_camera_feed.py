# video_client.py
# Run this on any machine on the ESP32 network to view the Jetson's
# camera feed. Connects to the Jetson directly (TCP, not broadcast) and
# reads length-prefixed JPEG frames. Standalone script -- just run it.

import socket
import struct

import cv2
import numpy as np

JETSON_IP = '192.168.4.2'
PORT = 5007
_HEADER = struct.Struct('<I')


def _recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError('server closed the connection')
        buf += chunk
    return buf


def main(ip=JETSON_IP, port=PORT):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print('Connected to video server at', (ip, port))

    try:
        while True:
            (length,) = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
            jpeg_bytes = _recv_exact(sock, length)

            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imshow('Jetson feed', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        sock.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
