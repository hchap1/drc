from machine import Pin, PWM
from time import sleep

# RIGHT MOTOR
m2_rpwm = PWM(Pin(18), freq=1000)
m2_lpwm = PWM(Pin(19), freq=1000)
m2_ren = Pin(21, Pin.OUT)
m2_len = Pin(22, Pin.OUT)
m2_ren.value(1)
m2_len.value(1)
m2_rpwm.duty(0)
m2_lpwm.duty(200)

sleep(0.1)

# LEFT MOTOR
m1_rpwm = PWM(Pin(25), freq=1000)
m1_lpwm = PWM(Pin(26), freq=1000)
m1_ren = Pin(27, Pin.OUT)
m1_len = Pin(14, Pin.OUT)
m1_ren.value(1)
m1_len.value(1)
m1_rpwm.duty(200)
m1_lpwm.duty(0)


print(m1_rpwm)
print(m1_lpwm)
print(m2_rpwm)
print(m2_lpwm)



sleep(15)

m1_rpwm.duty(0)
m2_lpwm.duty(0)
m1_ren.value(0)
m1_len.value(0)
m2_ren.value(0)
m2_len.value(0)

