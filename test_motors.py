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
        #
        # One side effect of connect()-ing a UDP socket: if the OS gets an
        # ICMP "port unreachable" back (e.g. the server isn't running /
        # rebooting), the *next* send() raises ConnectionRefusedError. That's
        # just diagnostic noise here, not a real failure -- UDP has no
        # actual connections to refuse, and a packet that doesn't make it
        # through is already the expected/handled case server-side. Swallow
        # it so a momentarily-absent server never crashes the caller.
        try:
            self._sock.send(packet)
        except OSError:
            pass

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
    # quick manual test
    import time

    c = connect()
    for i in range(10):
        c.send(0.2, 0.2)
        time.sleep(0.1)
    c.send(0, 0)
    c.close()
