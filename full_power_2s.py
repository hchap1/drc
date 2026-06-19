from motors import Motors
from time import sleep

motors = Motors()
motors.setPower(1, 1)
time.sleep(2)
motors.setPower(0, 0)
