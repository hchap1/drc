# ESP32 boot.py — runs before main.py on every power-on.
# Nothing to configure here; all setup (WiFi, UDP socket) is done in server.py.
#
# Network topology:
#   Jetson  — WiFi AP  — 192.168.4.1  (SSID: drc, password: drcpass)
#   ESP32   — STA mode — 192.168.4.2  (static, set in server.py)
#   Laptop  — STA mode — DHCP         (connects to same hotspot for collect_data.py)
#
# Motor commands flow:
#   Jetson → UDP → 192.168.4.2:5005 → ESP32
#
# To set up the Jetson hotspot (run once):
#   sudo nmcli device wifi hotspot ifname wlan0 ssid jetson-drc
