from machine import Pin, PWM

class Motors:
    MAX_DUTY = 1023

    def __init__(self):

        # Left
        self.m1_rpwm = PWM(Pin(25), freq=1000)
        self.m1_lpwm = PWM(Pin(26), freq=1000)
        self.m1_ren  = Pin(27, Pin.OUT)
        self.m1_len  = Pin(14, Pin.OUT)

        # Right
        self.m2_rpwm = PWM(Pin(18), freq=1000)
        self.m2_lpwm = PWM(Pin(19), freq=1000)
        self.m2_ren  = Pin(21, Pin.OUT)
        self.m2_len  = Pin(22, Pin.OUT)

        self.setPower(0, 0)

    def _set_motor(self, rpwm, lpwm, ren, len_pin, power):
        power = max(-1.0, min(1.0, power))
        duty  = int(abs(power) * self.MAX_DUTY)

        if power > 0:
            ren.value(1)
            len_pin.value(1)
            rpwm.duty(duty)
            lpwm.duty(0)
        elif power < 0:
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
        self._set_motor(self.m1_rpwm, self.m1_lpwm, self.m1_ren, self.m1_len, left)
        self._set_motor(self.m2_rpwm, self.m2_lpwm, self.m2_ren, self.m2_len,-right)

if __name__ == "__main__":
    from time import sleep
    motors = Motors()
    
    for i in range(1, 11):
        motors.setPower(i / 10, i / 10)
        sleep(0.2)

    motors.setPower(0, 0)
