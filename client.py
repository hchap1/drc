# client.py
# Plain Python (PC / laptop / Pi side). Sends fire-and-forget UDP power
# commands to the ESP32 motor server. No receiving is implemented since
# the server never replies.

import socket
import struct

DEFAULT_PORT = 5005


class Client:
    def __init__(self, sock: socket.socket, addr: tuple):
        self._sock = sock
        self._addr = addr

    def send(self, left: float, right: float) -> None:
        """Send a power update. left/right are clamped to [-1.0, 1.0]."""
        left = max(-1.0, min(1.0, left))
        right = max(-1.0, min(1.0, right))
        packet = struct.pack('<hh', int(left * 1000), int(right * 1000))
        # socket is "connected" (see connect() below) so we can use send()
        # instead of sendto() -- the kernel already has the destination
        # cached, which shaves a little off the per-call overhead.
        self._sock.send(packet)

    def close(self) -> None:
        self._sock.close()


def connect(ip: str = '192.168.4.1', port: int = DEFAULT_PORT) -> Client:
    """Create a UDP 'connection' to the motor server. Returns a Client
    with a .send(left, right) method. There's no handshake -- UDP
    connect() just fixes the destination address on this socket and lets
    us call send() instead of sendto()."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect((ip, port))
    return Client(sock, (ip, port))


if __name__ == '__main__':
    import time

    c = connect()
    c.send(0.5, 0.5)
    time.sleep(1)
    c.send(0, 0)
    c.close()
