# serial_client.py
# Jetson-side motor client. Writes motor packets directly to the USB CDC
# serial device as a plain file — no pyserial needed. USB CDC ignores baud
# rate at the USB level so no termios configuration is required.

import struct

DEFAULT_PORT = '/dev/ttyUSB0'


class SerialClient:
    def __init__(self, port: str):
        self._f = open(port, 'wb', buffering=0)

    def send(self, left: float, right: float) -> None:
        left  = max(-1.0, min(1.0, left))
        right = max(-1.0, min(1.0, right))
        self._f.write(struct.pack('<hh', int(left * 1000), int(right * 1000)))

    def send_raw(self, data: bytes) -> None:
        self._f.write(data)

    def close(self) -> None:
        self._f.close()


def connect(port: str = DEFAULT_PORT) -> SerialClient:
    return SerialClient(port)
