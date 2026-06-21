from machine import Pin, PWM
from time import sleep

m2_rpwm = PWM(Pin(18), freq=1100)
m2_lpwm = PWM(Pin(19), freq=1200)
m2_ren = Pin(21, Pin.OUT)
m2_len = Pin(22, Pin.OUT)

m1_rpwm = PWM(Pin(25), freq=1000)
m1_lpwm = PWM(Pin(26), freq=1300)
m1_ren = Pin(27, Pin.OUT)
m1_len = Pin(14, Pin.OUT)

m2_ren.value(1)
m2_len.value(1)
m1_ren.value(1)
m1_len.value(1)

print(m1_rpwm, m2_rpwm, m1_lpwm, m2_lpwm)

m1_rpwm.duty(100)
print(m1_rpwm, m2_rpwm, m1_lpwm, m2_lpwm)
m2_rpwm.duty(0)
print(m1_rpwm, m2_rpwm, m1_lpwm, m2_lpwm)
m2_lpwm.duty(100)
print(m1_rpwm, m2_rpwm, m1_lpwm, m2_lpwm)
m1_lpwm.duty(0)
print(m1_rpwm, m2_rpwm, m1_lpwm, m2_lpwm)

sleep(3)

m2_rpwm.duty(0)
m2_lpwm.duty(0)
m2_ren.value(0)
m2_len.value(0)

sleep(2)


m1_rpwm.duty(0)
m1_lpwm.duty(0)
m1_ren.value(0)
m1_len.value(0)
