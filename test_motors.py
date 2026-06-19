from machine import Pin, PWM
import time

m1_rpwm = PWM(Pin(25), freq=1000)
m1_lpwm = PWM(Pin(26), freq=1000)
m1_ren = Pin(27, Pin.OUT)
m1_len = Pin(14, Pin.OUT)

def stop_all():
    m1_rpwm.duty(0)
    m1_lpwm.duty(0)
    m1_ren.value(0)
    m1_len.value(0)

def full_power():
    m1_ren.value(1)
    m1_len.value(1)

    m1_rpwm.duty(100)
    m1_lpwm.duty(0)

stop_all()
time.sleep(1)

print("FULL POWER!")

full_power()

time.sleep(3)

print("STOPPING")
stop_all()
