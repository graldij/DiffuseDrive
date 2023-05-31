import numpy as np
from collections import deque

class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = self._window[-1] - self._window[-2]
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


class DiffuseDriveController(object):
    def __init__(self, config):
        self.turn_controller = PIDController(
            K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n
        )
        self.speed_controller = PIDController(
            K_P=config.speed_KP,
            K_I=config.speed_KI,
            K_D=config.speed_KD,
            n=config.speed_n,
        )
        
        self.config = config
        self.stop_steps = 0
        self.forced_forward_steps = 0

    def run_step(
        self, speed, waypoints):
        """
        speed: int, m/s
        waypoints: [float lits], 10 * 2, m
        """

        # control steering
        # breakpoint()
        aim = (waypoints[1] + waypoints[0]) / 2.0
        aim[1] *= -1
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        if speed < 0.01:
            angle = 0
        
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)


    
        # adapted from Transfuser
        # control speed 
        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
        brake = ((desired_speed < self.config.brake_speed) or ((speed / desired_speed) > self.config.brake_ratio))

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.max_throttle)
        
        if brake:
            angle = 0.0

        meta_info_1 = "speed: %.2f, target_speed: %.2f" % (
            speed,
            desired_speed,
        )

        return steer, throttle, brake, meta_info_1
