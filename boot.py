# This file is executed on every boot (including wake-boot from deepsleep)
#import esp
#esp.osdebug(None)
import network, webrepl

ap = network.WLAN(network.AP_IF)
ap.active(True)
ap.config(essid='drc-esp', password='drc123')

while not ap.active():
    pass

print("AP ready:", ap.ifconfig())

webrepl.start(password='drc123')
