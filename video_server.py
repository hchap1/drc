# video_server.py
# Runs on the Jetson Nano (192.168.4.2). Broadcasts JPEG-encoded frames
# over UDP so any listener on the network (running video_client.py) can
# reconstruct the feed. Same philosophy as client.py: fire-and-forget,
# a dropped or incomplete frame is just skipped, never retried/buffered.

import socket
import struct

import cv2

PORT = 5007
BROADCAST_ADDR = '192.168.4.255'   # subnet broadcast for the 192.168.4.0/24 ESP32 AP network
CHUNK_SIZE = 1400                  # stay under the typical 1500-byte MTU once headers are added
JPEG_QUALITY = 80
HEARTBEAT_EVERY = 30                # print a liveness line every N frames (0 to disable)

_HEADER = struct.Struct('<HHH')    # frame_id (uint16), chunk_index (uint16), total_chunks (uint16)


class VideoBroadcaster:
    def __init__(self, port=PORT, broadcast_addr=BROADCAST_ADDR, jpeg_quality=JPEG_QUALITY,
                 heartbeat_every=HEARTBEAT_EVERY):
        self._addr = (broadcast_addr, port)
        self._quality = jpeg_quality
        self._frame_id = 0
        self._heartbeat_every = heartbeat_every
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    def send(self, image) -> None:
        """JPEG-encode `image` and broadcast it in MTU-sized chunks."""
        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if not ok:
            return
        data = encoded.tobytes()

        total_chunks = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE
        if total_chunks > 65535:
            # frame is too big for the 2-byte chunk-count field (>90MB) --
            # essentially unreachable at sane resolutions/quality, but drop
            # rather than send something the client can't reassemble
            print('video_server: frame too large to send (%d bytes), dropping' % len(data))
            return

        frame_id = self._frame_id
        self._frame_id = (self._frame_id + 1) % 65536

        for i in range(total_chunks):
            chunk = data[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
            packet = _HEADER.pack(frame_id, i, total_chunks) + chunk
            self._sock.sendto(packet, self._addr)

        if self._heartbeat_every and frame_id % self._heartbeat_every == 0:
            print('video_server: sent frame %d (%d bytes, %d chunks) to %s'
                  % (frame_id, len(data), total_chunks, self._addr))

    def close(self) -> None:
        self._sock.close()


def broadcast(port=PORT, broadcast_addr=BROADCAST_ADDR, jpeg_quality=JPEG_QUALITY) -> VideoBroadcaster:
    return VideoBroadcaster(port=port, broadcast_addr=broadcast_addr, jpeg_quality=jpeg_quality)
