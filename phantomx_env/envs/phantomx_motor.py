import collections
import numpy as np

NUM_MOTORS = 18
CONTROL_MODES = ["TORQUE", "PD", "PID"]

class PhantomxMotorModel(object):
    def __init__(self, kp=60, kd=1, ki=0.1, torque_limits=None, motor_control_mode="PD"):
        self._kp = kp
        self._kd = kd
        self._ki = ki
        self._torque_limits = torque_limits
        self._motor_control_mode = motor_control_mode
        self._strength_ratios = np.full(NUM_MOTORS, 1)
        self._integral_error = np.zeros(NUM_MOTORS)  # Initialize integral error for PID
        if torque_limits is not None:
            if isinstance(torque_limits, (collections.abc.Sequence, np.ndarray)):
              self._torque_limits = np.asarray(torque_limits)
            else:
              self._torque_limits = np.full(NUM_MOTORS, torque_limits)

    def convert_to_torque(self, motor_commands, motor_angle, motor_velocity, motor_control_mode=None):
        if not motor_control_mode:
            motor_control_mode = self._motor_control_mode
        # print("motor_control_mode", motor_control_mode)
        if motor_control_mode == "TORQUE":
            assert len(motor_commands) == NUM_MOTORS
            motor_torques = self._strength_ratios * motor_commands
            motor_torques = np.clip(motor_torques, -1.0 * self._torque_limits, self._torque_limits)
            return motor_torques, motor_torques

        desired_motor_angles = None
        desired_motor_velocities = None
        kp = None
        kd = None
        ki = None
        additional_torques = np.full(NUM_MOTORS, 0)

        if motor_control_mode == "PD":
            assert len(motor_commands) == NUM_MOTORS
            kp = self._kp
            kd = self._kd
            ki = 0
            desired_motor_angles = motor_commands
            desired_motor_velocities = np.full(NUM_MOTORS, 0)
        elif motor_control_mode == "PID":
            assert len(motor_commands) == NUM_MOTORS
            kp = self._kp
            kd = self._kd
            ki = self._ki
            desired_motor_angles = motor_commands
            desired_motor_velocities = np.full(NUM_MOTORS, 0)
            # Calculate integral of error
            error = motor_angle - desired_motor_angles
            self._integral_error += error
            print("Integral error: ", self._integral_error)
        else:
            raise ValueError("Motor model should only be torque, PD, or PID control.")

        motor_torques = -1 * (kp * (motor_angle - desired_motor_angles) +
                              kd * (motor_velocity - desired_motor_velocities) +
                              ki * self._integral_error) + additional_torques
        motor_torques = self._strength_ratios * motor_torques
        if self._torque_limits is not None:
            if len(self._torque_limits) != len(motor_torques):
                raise ValueError("Torque limits dimension does not match the number of motors.")
            motor_torques = np.clip(motor_torques, -1.0 * self._torque_limits, self._torque_limits)
        return motor_torques, motor_torques