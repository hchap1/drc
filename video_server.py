# video_server.py
# Runs on the Jetson Nano (192.168.4.2). Accepts any number of TCP
# clients and pushes JPEG frames to each, length-prefixed. A frame
# either arrives whole or the connection stalls/retransmits -- no more
# manual chunk reassembly. A slow or dead client gets dropped (short
# send timeout) without holding up the others or the capture loop.

import socket
import struct

import cv2

PORT = 5007
JPEG_QUALITY = 70
STREAM_WIDTH = 640          # frames are downscaled to this width before sending (None = no resize)
SEND_TIMEOUT = 0.2          # max time to wait on one client's send before dropping it
HEARTBEAT_EVERY = 30        # print a liveness line every N frames (0 to disable)

_HEADER = struct.Struct('<I')  # 4-byte frame length prefix


class VideoServer:
    def __init__(self, port=PORT, jpeg_quality=JPEG_QUALITY, stream_width=STREAM_WIDTH,
                 send_timeout=SEND_TIMEOUT, heartbeat_every=HEARTBEAT_EVERY):
        self._quality = jpeg_quality
        self._stream_width = stream_width
        self._send_timeout = send_timeout
        self._heartbeat_every = heartbeat_every
        self._frame_count = 0
        self._clients = []  # connected client sockets

        self._listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listen_sock.bind(('0.0.0.0', port))
        self._listen_sock.listen(8)
        self._listen_sock.setblocking(False)  # accept() must never block the capture loop
        print('video_server: listening on port', port)

    def _accept_new_clients(self):
        while True:
            try:
                conn, addr = self._listen_sock.accept()
            except BlockingIOError:
                break
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(self._send_timeout)
            self._clients.append(conn)
            print('video_server: client connected from', addr)

    def send(self, image) -> None:
        self._accept_new_clients()
        if not self._clients:
            return  # nobody watching -- skip the encode work entirely

        if self._stream_width and image.shape[1] != self._stream_width:
            scale = self._stream_width / image.shape[1]
            new_size = (self._stream_width, int(round(image.shape[0] * scale)))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if not ok:
            return
        data = encoded.tobytes()
        packet = _HEADER.pack(len(data)) + data

        still_connected = []
        for conn in self._clients:
            try:
                conn.sendall(packet)
                still_connected.append(conn)
            except (OSError, socket.timeout):
                # client is gone or too slow to keep up -- drop it rather
                # than let it block the capture loop or the other clients
                try:
                    conn.close()
                except OSError:
                    pass
                print('video_server: client disconnected')
        self._clients = still_connected

        self._frame_count += 1
        if self._heartbeat_every and self._frame_count % self._heartbeat_every == 0:
            print('video_server: sent frame %d (%d bytes) to %d client(s)'
                  % (self._frame_count, len(data), len(self._clients)))

    def close(self) -> None:
        for conn in self._clients:
            try:
                conn.close()
            except OSError:
                pass
        self._listen_sock.close()


def serve(port=PORT, jpeg_quality=JPEG_QUALITY, stream_width=STREAM_WIDTH) -> VideoServer:
    return VideoServer(port=port, jpeg_quality=jpeg_quality, stream_width=stream_width)
