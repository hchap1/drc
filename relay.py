# relay.py
# Runs on the Jetson. Listens for UDP motor commands from the laptop on
# port 5005 and forwards the raw bytes to the ESP32 over USB serial.
# Run this for manual control (pygame_control.py on the laptop).

import socket
import serial_client

UDP_PORT = 5005


def main():
    motors = serial_client.connect()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', UDP_PORT))
    sock.settimeout(0.5)

    print(f'Relay: UDP:{UDP_PORT} → ESP32 via serial')

    while True:
        try:
            data, _ = sock.recvfrom(16)
            if len(data) == 4:
                motors.send_raw(data)
        except socket.timeout:
            pass


if __name__ == '__main__':
    main()
