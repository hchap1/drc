# motor_client.py
# Jetson-side motor client.  Sends UDP packets to the ESP32.
# UDP is message-based so there is no byte-stream framing or OS line-discipline
# mangling risk.  Each send() delivers exactly one atomic 4-byte packet.
#
# Packet format: 4 bytes, little-endian signed int16 pair <hh
#   left_power  = int16 / 1000.0
#   right_power = int16 / 1000.0

import socket
import struct

ESP32_IP   = '192.168.4.2'   # static IP assigned in server.py
ESP32_PORT = 5005

_PACK = struct.Struct('<hh')


class MotorClient:
    def __init__(self, ip: str, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (ip, port)

    def send(self, left: float, right: float) -> None:
        left  = max(-1.0, min(1.0, left))
        right = max(-1.0, min(1.0, right))
        try:
            self._sock.sendto(_PACK.pack(int(left * 1000), int(right * 1000)), self._addr)
        except OSError:
            pass   # UDP fire-and-forget — a dropped packet is not fatal

    def close(self) -> None:
        self._sock.close()


def connect(ip: str = ESP32_IP, port: int = ESP32_PORT) -> MotorClient:
    return MotorClient(ip, port)
