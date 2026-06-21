from motors import Motors
from time import sleep

motors = Motors()
motors.setPower(0.2, 0.2)
sleep(1)
motors.setPower(0, 0)

