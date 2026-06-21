import network, webrepl

ap = network.WLAN(network.AP_IF)
ap.active(True)
ap.config(essid='drc-esp', password='drc123')

while not ap.active():
    pass

print("AP ready:", ap.ifconfig())

webrepl.start(password='drc123')
