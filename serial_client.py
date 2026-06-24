# serial_client.py
# Jetson-side motor client. Sends fire-and-forget power commands to the ESP32
# over USB serial. Drop-in replacement for client.py — same connect()/send()/close() API.

import struct
import serial

DEFAULT_PORT = '/dev/ttyUSB0'
DEFAULT_BAUD = 115200


class SerialClient:
    def __init__(self, port: str, baud: int):
        self._ser = serial.Serial(port, baud, timeout=1)

    def send(self, left: float, right: float) -> None:
        left  = max(-1.0, min(1.0, left))
        right = max(-1.0, min(1.0, right))
        self._ser.write(struct.pack('<hh', int(left * 1000), int(right * 1000)))

    def send_raw(self, data: bytes) -> None:
        self._ser.write(data)

    def close(self) -> None:
        self._ser.close()


def connect(port: str = DEFAULT_PORT, baud: int = DEFAULT_BAUD) -> SerialClient:
    return SerialClient(port, baud)
