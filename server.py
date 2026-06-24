# server.py
# Runs on the ESP32 (MicroPython). Reads motor power commands from the USB
# serial connection (the same port used for flashing/REPL) and drives the
# motors directly. A 4-byte little-endian packet (<hh) encodes left and
# right motor power as integers scaled by 1000.
#
# To run on boot, copy this onto the device as main.py (or import it
# from your existing main.py / boot.py).

import sys
import uselect
import struct
import time
from machine import Pin, PWM

WATCHDOG_TIMEOUT_MS = 500   # if no packet arrives for this long, stop motors
POLL_TIMEOUT_MS     = 100   # poll interval when idle (for watchdog checking)


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

    poll = uselect.poll()
    poll.register(sys.stdin, uselect.POLLIN)

    print('Motor serial server ready')

    last_rx = time.ticks_ms()
    stopped = True
    buf = b''

    while True:
        events = poll.poll(POLL_TIMEOUT_MS)

        if events:
            chunk = sys.stdin.buffer.read(1)
            if chunk:
                buf += chunk
                while len(buf) >= 4:
                    left_i, right_i = struct.unpack('<hh', buf[:4])
                    buf = buf[4:]
                    motors.setPower(left_i / 1000.0, right_i / 1000.0)
                    last_rx = time.ticks_ms()
                    stopped = False
        else:
            # Poll timed out — check watchdog
            if not stopped and time.ticks_diff(time.ticks_ms(), last_rx) > WATCHDOG_TIMEOUT_MS:
                motors.setPower(0, 0)
                stopped = True


if __name__ == '__main__':
    run()
