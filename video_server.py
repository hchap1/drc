import queue
import socket
import struct
import threading

import cv2

PORT = 5007
JPEG_QUALITY = 70
STREAM_WIDTH = 240
HEARTBEAT_EVERY = 30

_HEADER = struct.Struct('<I')


class _Client:
    def __init__(self, conn, addr):
        self.conn = conn
        self.addr = addr

        # Unlimited backlog.
        self.queue = queue.Queue()

        self.thread = threading.Thread(
            target=self._sender_loop,
            daemon=True,
        )

        self.thread.start()

    def _sender_loop(self):
        try:
            while True:
                packet = self.queue.get()

                if packet is None:
                    break

                self.conn.sendall(packet)

        except OSError:
            pass

        try:
            self.conn.close()
        except OSError:
            pass

    def send(self, packet):
        self.queue.put(packet)

    def close(self):
        self.queue.put(None)


class VideoServer:
    def __init__(
        self,
        port=PORT,
        jpeg_quality=JPEG_QUALITY,
        stream_width=STREAM_WIDTH,
        heartbeat_every=HEARTBEAT_EVERY,
    ):
        self._quality = jpeg_quality
        self._stream_width = stream_width
        self._heartbeat_every = heartbeat_every

        self._frame_count = 0
        self._clients = []

        self._listen_sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )

        self._listen_sock.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )

        self._listen_sock.bind(('0.0.0.0', port))
        self._listen_sock.listen(8)
        self._listen_sock.setblocking(False)

        print('video_server: listening on port', port)

    def _accept_new_clients(self):
        while True:
            try:
                conn, addr = self._listen_sock.accept()
            except BlockingIOError:
                break

            conn.setsockopt(
                socket.IPPROTO_TCP,
                socket.TCP_NODELAY,
                1,
            )

            client = _Client(conn, addr)

            self._clients.append(client)

            print(
                'video_server: client connected from',
                addr,
            )

    def send(self, image):
        self._accept_new_clients()

        if not self._clients:
            return

        if (
            self._stream_width
            and image.shape[1] != self._stream_width
        ):
            scale = self._stream_width / image.shape[1]

            new_size = (
                self._stream_width,
                int(round(image.shape[0] * scale)),
            )

            image = cv2.resize(
                image,
                new_size,
                interpolation=cv2.INTER_AREA,
            )

        ok, encoded = cv2.imencode(
            '.jpg',
            image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                self._quality,
            ],
        )

        if not ok:
            return

        data = encoded.tobytes()

        packet = (
            _HEADER.pack(len(data))
            + data
        )

        for client in self._clients:
            client.send(packet)

        self._frame_count += 1

        if (
            self._heartbeat_every
            and self._frame_count % self._heartbeat_every == 0
        ):
            print(
                'video_server: sent frame %d (%d bytes) to %d client(s)'
                % (
                    self._frame_count,
                    len(data),
                    len(self._clients),
                )
            )

    def close(self):
        for client in self._clients:
            client.close()

        self._listen_sock.close()


def serve(
    port=PORT,
    jpeg_quality=JPEG_QUALITY,
    stream_width=STREAM_WIDTH,
):
    return VideoServer(
        port=port,
        jpeg_quality=jpeg_quality,
        stream_width=stream_width,
    )
