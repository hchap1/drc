from machine import Pin, PWM

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
        self.right_lpwm = PWM(Pin(19), freq=1300)
        self.right_ren  = Pin(21, Pin.OUT)
        self.right_len  = Pin(22, Pin.OUT)

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
        self._set_motor(self.left_rpwm, self.left_lpwm, self.left_ren, self.left_len, left)
        self._set_motor(self.right_rpwm, self.right_lpwm, self.right_ren, self.right_len,-right)
