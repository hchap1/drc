# video_client.py
# Run this on any machine on the ESP32 network to view the Jetson's
# camera feed broadcast by video_server.py. This is a standalone script,
# not a library -- just run it directly.

import socket
import struct

import cv2
import numpy as np

PORT = 5007
_HEADER = struct.Struct('<HHH')
HEADER_SIZE = _HEADER.size


def main(port=PORT):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', port))

    print('Listening for video broadcast on port', port)

    current_frame_id = None
    chunks = {}
    expected_total = 0
    chunks_received = 0

    try:
        while True:
            packet, addr = sock.recvfrom(65535)
            if len(packet) < HEADER_SIZE:
                continue

            chunks_received += 1
            if chunks_received % 100 == 1:
                print('video_client: received %d chunks so far (latest from %s)' % (chunks_received, addr))

            frame_id, chunk_index, total_chunks = _HEADER.unpack_from(packet, 0)
            payload = packet[HEADER_SIZE:]

            if frame_id != current_frame_id:
                # A new frame has started arriving -- whatever was
                # buffered for the old one is stale (a chunk got lost),
                # so drop it instead of waiting around for it.
                current_frame_id = frame_id
                chunks = {}
                expected_total = total_chunks

            chunks[chunk_index] = payload

            if len(chunks) == expected_total:
                jpeg_bytes = b''.join(chunks[i] for i in range(expected_total))
                arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    cv2.imshow('Jetson feed', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                current_frame_id = None
                chunks = {}
    finally:
        sock.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
