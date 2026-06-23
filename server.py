# server.py
# Runs on the ESP32 (MicroPython). Listens for UDP power commands and
# drives the motors directly from the receive loop -- no queues, no
# polling, no thread handoff. A packet arriving is the only thing that
# triggers work.
#
# To run on boot, copy this onto the device as main.py (or import it
# from your existing main.py / boot.py).

import socket
import struct
import time
from machine import Pin, PWM

PORT = 5005
WATCHDOG_TIMEOUT_MS = 500   # if no packet arrives for this long, stop motors
SOCK_TIMEOUT_S = 0.2        # how often we wake up (when idle) to check the watchdog


class Motors:
    MAX_DUTY = 1023

    def __init__(self):
        # Left
        self.left_rpwm = PWM(Pin(25), freq=1000)
        self.left_lpwm = PWM(Pin(26), freq=1100)
        self.left_ren  = Pin(27, Pin.OUT)
        self.left_len  = Pin(14, Pin.OUT)
        # Right
        self.right_rpwm = PWM(Pin(18), freq=1200)
        self.right_lpwm = PWM(Pin(19), freq=1000)
        self.right_ren  = Pin(21, Pin.OUT)
        self.right_len  = Pin(22, Pin.OUT)
        self.setPower(0, 0)

    def _set_motor(self, rpwm, lpwm, ren, len_pin, power):
        power = max(-1.0, min(1.0, power))
        duty  = int(abs(power) * self.MAX_DUTY)
        if power > 0.05:
            ren.value(1)
            len_pin.value(1)
            rpwm.duty(duty)
            lpwm.duty(0)
        elif power < -0.05:
            ren.value(1)
            len_pin.value(1)
            rpwm.duty(0)
            lpwm.duty(duty)
        else:
            ren.value(0)
            len_pin.value(0)
            rpwm.duty(0)
            lpwm.duty(0)

    def setPower(self, left, right):
        self._set_motor(self.left_rpwm, self.left_lpwm, self.left_ren, self.left_len, left)
        self._set_motor(self.right_rpwm, self.right_lpwm, self.right_ren, self.right_len, -right)


def run():
    motors = Motors()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except OSError:
        pass  # not all MicroPython socket builds support this option
    sock.bind(('0.0.0.0', PORT))

    # Blocking recvfrom gives the lowest latency response to an arriving
    # packet (no polling interval). The timeout only matters when NO data
    # is arriving -- it's how we periodically check the watchdog.
    sock.settimeout(SOCK_TIMEOUT_S)

    print('Motor UDP server listening on port', PORT)

    last_rx = time.ticks_ms()
    stopped = True

    while True:
        try:
            data, addr = sock.recvfrom(16)
        except OSError:
            # Recv timed out -- no packet arrived in SOCK_TIMEOUT_S.
            # This is also our only signal that a client may have vanished,
            # since UDP has no disconnect event.
            if not stopped and time.ticks_diff(time.ticks_ms(), last_rx) > WATCHDOG_TIMEOUT_MS:
                motors.setPower(0, 0)
                stopped = True
            continue

        if len(data) != 4:
            continue  # malformed/foreign packet, ignore and keep listening

        try:
            left_i, right_i = struct.unpack('<hh', data)
        except Exception:
            continue

        motors.setPower(left_i / 1000.0, right_i / 1000.0)
        last_rx = time.ticks_ms()
        stopped = False
