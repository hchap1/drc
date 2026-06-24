# relay.py
# Runs on the Jetson during manual driving (pygame_control.py on the laptop).
# Receives UDP motor packets from the laptop on port 5005 and forwards them
# to the ESP32 via motor_client (UDP to 192.168.4.2:5005).

import socket
import struct

import motor_client

UDP_PORT = 5005
_PACK    = struct.Struct('<hh')


def main():
    motors = motor_client.connect()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', UDP_PORT))
    sock.settimeout(0.5)

    print(f'Relay: laptop UDP:{UDP_PORT} → ESP32 {motor_client.ESP32_IP}:{motor_client.ESP32_PORT}')

    try:
        while True:
            try:
                data, _ = sock.recvfrom(16)
                if len(data) == 4:
                    left_i, right_i = _PACK.unpack(data)
                    motors.send(left_i / 1000.0, right_i / 1000.0)
            except socket.timeout:
                pass
    finally:
        motors.send(0.0, 0.0)
        motors.close()
        sock.close()


if __name__ == '__main__':
    main()
