# server.py — ESP32 motor server (MicroPython)
#
# Connects to the Jetson's WiFi hotspot in station mode, assigns itself a
# static IP, then listens on UDP port 5005 for motor command packets sent
# by the Jetson.  Reconnects automatically if WiFi drops.
#
# Packet format: 4 bytes, little-endian signed int16 pair <hh
#   left_power  = int16 / 1000.0   range [-1.0, 1.0]
#   right_power = int16 / 1000.0   range [-1.0, 1.0]
#
# UDP is message-based so there is no byte-stream framing risk.
# Each recvfrom() delivers exactly one atomic 4-byte packet.

import network
import socket
import struct
import time
from machine import Pin, PWM

# ── Configuration — edit these to match your setup ───────────────────────────
WIFI_SSID     = 'drc-jetson'
WIFI_PASSWORD = ''        # set to '' for an open (no-password) AP
STATIC_IP     = '192.168.4.2'   # fixed address so the Jetson always knows where to send
NETMASK       = '255.255.255.0'
GATEWAY       = '192.168.4.1'   # Jetson hotspot address
DNS           = '192.168.4.1'
UDP_PORT      = 5005

WATCHDOG_MS   = 500             # stop motors if no valid packet for this long
SOCKET_TMO    = 0.1             # recvfrom timeout (s) — keeps the watchdog responsive


# ── Motors ────────────────────────────────────────────────────────────────────

class Motors:
    MAX_DUTY = 1023

    def __init__(self):
        self.left_rpwm  = PWM(Pin(25), freq=1000)
        self.left_lpwm  = PWM(Pin(26), freq=1100)
        self.left_ren   = Pin(27, Pin.OUT)
        self.left_len   = Pin(14, Pin.OUT)
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
        # right motor is mounted mirrored — invert so positive = forward for both
        self._set_motor(self.right_rpwm, self.right_lpwm, self.right_ren, self.right_len, -right)


# ── WiFi ──────────────────────────────────────────────────────────────────────

def _make_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', UDP_PORT))
    sock.settimeout(SOCKET_TMO)
    return sock


def _wifi_connect():
    """Block until WiFi is up. Can be called again after a drop."""
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    while True:
        if not wlan.isconnected():
            print('Connecting to', WIFI_SSID, '...')
            wlan.disconnect()
            time.sleep_ms(200)
            if WIFI_PASSWORD:
                wlan.connect(WIFI_SSID, WIFI_PASSWORD)
            else:
                wlan.connect(WIFI_SSID)
            deadline = time.ticks_add(time.ticks_ms(), 15000)
            while not wlan.isconnected():
                if time.ticks_diff(deadline, time.ticks_ms()) <= 0:
                    break
                time.sleep_ms(500)
        if wlan.isconnected():
            # Set static IP after DHCP so it is not overwritten by the handshake
            wlan.ifconfig((STATIC_IP, NETMASK, GATEWAY, DNS))
            print('WiFi up:', wlan.ifconfig())
            return wlan
        print('WiFi failed, retrying in 2 s...')
        time.sleep(2)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    motors = Motors()
    wlan   = _wifi_connect()
    sock   = _make_socket()

    print('Listening for motor commands on UDP port', UDP_PORT)

    last_rx = time.ticks_ms()
    stopped = True

    while True:
        # ── WiFi watchdog ─────────────────────────────────────────────────────
        if not wlan.isconnected():
            motors.setPower(0, 0)
            stopped = True
            print('WiFi lost, reconnecting...')
            try:
                sock.close()
            except OSError:
                pass
            wlan = _wifi_connect()
            sock = _make_socket()
            last_rx = time.ticks_ms()
            continue

        # ── Receive motor packet ──────────────────────────────────────────────
        try:
            data, _ = sock.recvfrom(8)
            if len(data) == 4:
                left_i, right_i = struct.unpack('<hh', data)
                motors.setPower(left_i / 1000.0, right_i / 1000.0)
                last_rx = time.ticks_ms()
                stopped = False
        except OSError:
            pass   # timeout — fall through to watchdog check

        # ── Motor watchdog ────────────────────────────────────────────────────
        if not stopped and time.ticks_diff(time.ticks_ms(), last_rx) > WATCHDOG_MS:
            motors.setPower(0, 0)
            stopped = True
            print('Watchdog: no packet for', WATCHDOG_MS, 'ms — motors stopped')


if __name__ == '__main__':
    run()
