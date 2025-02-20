import gym
from gym import spaces
from gym.utils import seeding

import pybullet
from pybullet_utils import bullet_client as bc
import pybullet_data

import numpy as np
import random
from gym.utils import seeding
from phantomx_env.envs.phantomxCPG_Trackline import Phantomx
import math
import time
import os
import sys
# sys.path.append('/home/yangzhe/Intern/simulation/RL_phantomx_pybullet/src')


from src.CentralPatternGenerators.Hpof import PhantomxCPG
from src.AssistModulesCode.MatPlotAssitor import PlotModuleAssistor
from src.AssistModulesCode.ActionSelector import ActionModuleSelector
from src.CentralPatternGenerators.OnlineCPG import OnlinePhantomxCPG
from src.AssistModulesCode.Euler import quaternion_to_euler

RENDER_HEIGHT = 360
RENDER_WIDTH = 480
NUM_MOTORS = 18
LISTLEN = 47
FREC = 100.0
REWARD_FACTOR = 0.25

current_path = os.getcwd()

class LimitedList:
    def __init__(self, limit):
        self.limit = limit
        self.data = [0] * limit

    def add_element(self, element):
        if len(self.data) >= self.limit:
            self.data.pop(0)  # 从头部删除第一个元素

        self.data.append(element)  # 添加新元素

    def calculate_sum(self):
        return sum(self.data)  # 计算列表内所有元素的和
    
    def clear(self):
        self.data = [0] * self.limit


class PhantomxGymEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 100}
	
    def __init__(self, urdf_root=pybullet_data.getDataPath(), 
                 render=False,
                 set_goal_flag=False,
                 distance_limit=3,
                 forward_reward_cap=float("inf"), 
                 dist_weight = 1.0, # 1
                 height_weight = 0.0,# 5
                 shakevel_weight = 0.0,# 5
                 energy_weight = 0.0,# 100
                 action_rate = 0.0, # 1
                 yaw_weight = 0.0, # 3
                 hard_reset=True,
                 phantomx_urdf_root = current_path + "/phantomx_description"):
                #  phantomx_urdf_root="/home/yangzhe/Intern/simulation/RL_phantomx_pybullet/phantomx_description"):
                # phantomx_urdf_root="/home/yangzhe/Intern/simulation/RL_phantomx_pybullet/hexapod_34/urdf"):
        super(PhantomxGymEnv, self).__init__()  
        self._urdf_root = urdf_root
        self._phantomx_urdf_root = phantomx_urdf_root
        # self._obs_urdf_root = "/home/yangzhe/Intern/simulation/RL_phantomx_pybullet/ObstacleReg/urdf"
        self._obs_urdf_root = current_path + "/ObstacleReg/urdf"
        self._observation = []
        self._norm_observation = []
        self._env_step_counter = 0
        self._course_counter = 0
        self._is_render = render
        self._set_goal_flag = set_goal_flag
        self._cam_dist = 2.0
        self._cam_yaw = 0
        self._cam_pitch = -30
        self._last_frame_time = 0.0
        self.control_time_step = 1.0 / FREC
        self._distance_limit = distance_limit
        self._forward_reward_cap = forward_reward_cap
        self._action_change_cap = 0
        self._time_step = 1.0 / FREC
        self._max_episode_steps = 1024
        self._velrewardlist = []
        self._last_reward = -np.inf
        
        self.initial_action = [-0.523599, -0.523599, -0.523599, 0.523599, 0.523599, 0.523599, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25]

        self.forward_reward = 0
        self.drift_reward = 0
        self.energy_reward = 0
        self.shake_reward = 0
        self.height_reward = 0
        
        self.forward_reward_list = LimitedList(LISTLEN * 3)
        self.drift_reward_list = LimitedList(LISTLEN * 3)
        self.energy_reward_list = LimitedList(LISTLEN * 3)
        self.shake_reward_list = LimitedList(LISTLEN * 3)
        self.height_reward_list = LimitedList(LISTLEN * 3)
        self.yaw_reward_list = LimitedList(LISTLEN * 3)

        # Goal State
        self._goal_posture = []
        self._goal_velocity = []
        self._goal_state = [0, 0, 0]#x y velocity direction and magnitude
        self.test_goal_state = [0, 0, 0]
        self._curr_state = [0, 0, 0]

        self._prev_pos_to_goal = 0
        
        # energy reward cal
        self._dt_motor_torques = []
        self._dt_motor_velocities = []

        self._periodData = []
        self._periodDataIndex = 0

        self._objective_weights = [dist_weight, height_weight, shakevel_weight, energy_weight, action_rate, yaw_weight]
        self._objectives = []
        if self._is_render:
            self._pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._pybullet_client = bc.BulletClient()
        
        # 机器人运动关节角初始值
        self.history_data = np.array([-1.08321732, -0.00918567,  1.08317018, -0.02320317, -1.08321732, -0.00918567,
                                      1.08317018, -0.02320317, -1.08321732, -0.00918567,  1.08317018, -0.02320317,
                                      0.28975204, -1.98700634,  0.28975204, -1.98700634, -0.13758287,  2.00311752,
                                      -0.13758287,  2.00311752,  0.28975204, -1.98700634,  0.28975204, -1.98700634,
                                      -0.13758287,  2.00311752, -0.13758287,  2.00311752,  0.28975204, -1.98700634,
                                      0.28975204, -1.98700634, -0.13758287,  2.00311752, -0.13758287,  2.00311752]).reshape(1, -1)
        self._data = self.history_data.reshape(1, -1)
        self._CPG_obs = np.array([0.29616917,  1.04204406, -0.37516158, -1.01642539]).reshape(1, -1)
        self._last_motor_angle = [0] * NUM_MOTORS
        self._collision = [0] * 6
        self._hard_reset = True
		
        self.seed()
        self.reset()

        # self._action_bound = 3.0
        self._action_bound = 1.0
        # action_dim = NUM_MOTORS
        action_dim = 12
        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high)

        # action rate
        self._last_action = [0] * action_dim
        self._delta_action = [0] * action_dim

        self._last_average_velocity = [0]*3
        self._last_velocity = [0]*3

        self.history_action = []

        self._motorcommand = []

        observation_high = self._get_observation_upper_bound()
        # observation_low = -observation_high
        observation_low = self._get_observation_lower_bound()
        # print("observation_upper_bound:", observation_high)
        self.observation_space = spaces.Box(observation_low, observation_high)	 
        self._hard_reset = hard_reset

        self.PltModule = PlotModuleAssistor()
        self.CPGModule = PhantomxCPG()
        self.ActionModule = ActionModuleSelector()
        self.OnlineCPGModule = OnlinePhantomxCPG()

    def setMotorCommand(self, motorcommand):
        self._motorcommand = motorcommand

    def step(self, action):
        """Step forward the simulation, given the action.

        Args:
        action: A list of desired motor angles for motors.

        Returns:
          observations: 
          reward: The reward for the current state-action pair.
          done: Whether the episode has ended.
          info: A dictionary that stores diagnostic information.
        """
        # self._course_counter += 1
        # print("course_counter:", self._course_counter)
        self._last_base_position = self.phantomx.GetBasePosition()
        self._last_motor_angle = self.phantomx.GetTrueMotorAngles()
        # self._collision = self.phantomx.GetCollisionWithGround() 
        # print("collision:", self._collision)
        # print("goal_state", self._goal_state)
        # print("action: ", action)
        self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
                                                    self._cam_pitch, self.phantomx.GetBasePosition())
        
        # time_spent = time.time() - self._last_frame_time
        # self._last_frame_time = time.time()
        # time_to_sleep = self.control_time_step - time_spent
        # if time_to_sleep > 0:
        #     time.sleep(time_to_sleep)

        # CPG步态计算
        t = np.linspace(self._env_step_counter*self._time_step, self._env_step_counter*self._time_step+self._time_step, 2)
        # data = self.OnlineCPGModule.online_calculate(t, initial_values=self.history_data[-1, :])
        data = self._periodData[self._periodDataIndex]
        if self._periodDataIndex < len(self._periodData)-1:
            self._periodDataIndex += 1
        else:
            self._periodDataIndex = 0
        # print("data:", data)
        data.reshape(1, -1)
        # print("data:", data)
        if (len(data)==2):
            data = data[1:].reshape(1, -1)
        self._data = data
        # 取self._data 36个数据中的前四个
        self._CPG_obs = np.array([data[0][0], data[0][1], data[0][2], data[0][3]]).reshape(1, -1)
        # self._CPG_obs = np.array([data[0], data[1], data[2], data[3]]).reshape(1, -1)
        self.history_data = np.vstack((self.history_data, data))

        self._motorcommands = self.ActionModule.SelectAction(action_mode=10, t=t, data=data[-1, :])
        # self._motorcommands = self.ActionModule.SelectAction(action_mode=10, t=t, data=data)
        # print("DATA:", data)
        # print("HISTORY_DATA:", self.history_data)
        # print("MOTORCOMMANDS:", self._motorcommands)
        # CPG步态计算结束

        action = self._transform_action_to_motor_command(action)
        # print("motorcommand:", self._motorcommands)
        # print("action: ", action)
        motorcommand = self._transform_motor_to_motor_command(self._motorcommands, action)
        # action = self._transform_action_to_motor_command(action)
        # print("motorcommand:", motorcommand)
        self._dt_motor_torques = []
        self._dt_motor_velocities = []
        for i in range(3):
            self.phantomx.Step(motorcommand)
            self._dt_motor_torques.append(self.phantomx.GetTrueMotorTorques())
            self._dt_motor_velocities.append(self.phantomx.GetTrueMotorVelocities())
            self.forward_reward_list.add_element(self.phantomx.GetTrueBodyLinearVelocity()[0])
            self.drift_reward_list.add_element(self.phantomx.GetTrueBodyLinearVelocity()[1])
            self.yaw_reward_list.add_element(self.phantomx.GetTrueBodyAngularVelocity()[2])
        # self.phantomx.Step(motorcommand)
        # self.phantomx.Step(motorcommand)
        # print reward
        # print("reward:", reward)

        done = self._termination()
        # print("robotpos:", self.phantomx.GetBasePosition())
        # if done:
        #     print("is_done:", done)

        self.history_action = motorcommand        

        if done:
            self.phantomx.Terminate()
        
        observation = np.array(self._get_observation()).astype(np.float32)

        reward = self._reward(observation, action)

        self._prev_pos_to_goal, _ = self.get_distance_and_angle_to_goal(self._goal_state, self._curr_state)

        # action rate
        self._delta_action = np.array(action) - np.array(self._last_action)
        self._last_action = action

        info = {}

        self._env_step_counter += 1

        return observation, reward, done, info
        
    def reset(self):
	    #重新初始化
        
        # self._goal_state[0] = random.uniform(0.4, 0.6)
        self._goal_state[0] = 6
        self._goal_state[1] = 0
        self._goal_state[2] = 0
        if self._set_goal_flag:
            self._goal_state = self.test_goal_state
        self._last_action = [0] * 12
        self._delta_action = [0] * 12
        #关节角初始化
        self.history_data = np.array([0.29616917,  1.04204406, -0.37516158, -1.01642539,  0.29616917,  1.04204406,
                    -0.37516158, -1.01642539,  0.29616917,  1.04204406, -0.37516158, -1.01642539,
                    -1.97166912,  0.36889423, -1.97166912,  0.36889423,  1.86400333, -0.74193836,
                    1.86400333, -0.74193836, -1.97166912,  0.36889423, -1.97166912,  0.36889423,
                    1.86400333, -0.74193836,  1.86400333, -0.74193836, -1.97166912,  0.36889423,
                    -1.97166912, 0.36889423,  1.86400333, -0.74193836,  1.86400333, -0.74193836]).reshape(1, -1)
        
        self._periodData = []
        with open("history_data.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                self._periodData.append(np.array(eval(line)).reshape(1, -1))
        self._periodDataIndex = 0

        # print("reset===========================")
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 0)
        # print("reset client done")
        self.forward_reward_list.clear()
        self.drift_reward_list.clear()
        self.energy_reward_list.clear()
        self.shake_reward_list.clear()
        self.height_reward_list.clear()
        self.yaw_reward_list.clear()
        self._dt_motor_torques = []
        self._dt_motor_velocities = []
        self._prev_pos_to_goal, _ = self.get_distance_and_angle_to_goal(self._goal_state, self._curr_state) 
    
        if self._hard_reset:
            # print("reset simulation")
            self._pybullet_client.resetSimulation()
            # print("set gravity")
            self._pybullet_client.setGravity(0, 0, -10)
            # print("load plane")
            self._ground_id = self._pybullet_client.loadURDF("%s/plane.urdf" % self._urdf_root)
            # print("load phantomx")
            # self.ground_id = self._pybullet_client.loadURDF("plane.urdf")
            self.phantomx = Phantomx(pybullet_client=self._pybullet_client, urdf_root=self._phantomx_urdf_root)
            # print("reset phantomx")
            # self._obs1 = self._pybullet_client.loadURDF("%s/ObstacleReg.urdf" % self._obs_urdf_root, [-1,0,0.12], [0.707, 0, 0., 0.707], useFixedBase=True)
            self._pybullet_client.setTimeStep(1.0 / 1000)
        self.phantomx.Reset(reload_urdf=False)
        # print("reset done")
        self._last_motor_angle = self.phantomx.GetTrueMotorAngles()
        self._env_step_counter = 0
        self._last_base_position = [0, 0, 0.17]
        # print("reset debug")
        # self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
        #                                              self._cam_pitch, [0, 0, 0])
        self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
                                                         self._cam_pitch, self.phantomx.GetBasePosition())
        # print("reset done")
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 1)
        # print("reset done===========================")
        return np.array(self._get_observation())
		
    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])
        base_pos = self.phantomx.GetBasePosition()
        view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(fov=60,
                                                                       aspect=float(RENDER_WIDTH) /
                                                                       RENDER_HEIGHT,
                                                                       nearVal=0.1,
                                                                       farVal=100.0)
        (_, _, px, _, _) = self._pybullet_client.getCameraImage(
            width=RENDER_WIDTH,
            height=RENDER_HEIGHT,
            renderer=self._pybullet_client.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        self.phantomx.Terminate()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reward_function(self, desired_x, current_x, yaw_flag):
        # if yaw_flag:
        #     return np.e**(-abs(desired_x - current_x) / 1)
        # return np.e**(-abs(desired_x - current_x) / REWARD_FACTOR)
        if yaw_flag:
            if current_x == desired_x:
                return 1
            elif current_x < desired_x:
                return 0.1 * (current_x - desired_x) + 1
            else:
                return 0.1 * (-current_x + desired_x) + 1
        if current_x == desired_x:
            return 1
        elif current_x < desired_x:
            return 2 * (current_x - desired_x) + 1
        else:
            return 2 * (-current_x + desired_x) + 1
        # if yaw_flag:
        #     return -5*(current_x - desired_x)**2 + 1
        # return -15*(current_x - desired_x)**2 + 1
    
    def penalty_function(self, desired_x, current_x, angvel_flag):
        if angvel_flag:
            return -0.1 * abs(current_x - desired_x)
        return -2 * abs(current_x - desired_x)
        # return -abs(current_x - desired_x)

    def radio_function(self, desired_x, ratio, timestep):
        return desired_x / ratio**2 * timestep**2

    def get_distance_and_angle_to_goal(self, goal_state, current_state):
        # print("goal_state:", goal_state)
        # print("current_state:", current_state)
        goal_state = np.array(goal_state)
        current_state = np.array(current_state)
        distance = np.linalg.norm(goal_state[0:2] - current_state[0:2])
        angle = np.arctan2(goal_state[1] - current_state[1], goal_state[0] - current_state[0])
        # angle = goal_state[2] - current_state[2]
        return distance, angle


    # 只有x方向平均速度
    def _reward(self, observations, action):
        current_goal_state = observations[0:3] * 10
        current_CPG_data = observations[3:7] * 4
        current_orientation = observations[7:11] * 1
        current_base_velocity = observations[11:14] * 4
        current_base_angvelocity = observations[14:17] * 30
        current_joint_angles = observations[17:35] * 5.0
        current_base_position = observations[35:38] * 10
        current_joint_angvelocities = self.phantomx.GetTrueMotorVelocities()
        current_joint_torques = self.phantomx.GetTrueMotorTorques()
        last_action = observations[38:50]
        delta_action = action - last_action

        self._curr_state = [current_base_position[0], current_base_position[1], current_base_position[2]]
        
        curr_dist_to_goal, curr_angle_to_goal = self.get_distance_and_angle_to_goal(self._goal_state, self._curr_state)

        print(curr_dist_to_goal)
        # minize distance to goal (move towards the goal)
        dist_reward = (self._prev_pos_to_goal - curr_dist_to_goal) * 10
        # minize yaw deviation to the goal
        # angle_reward = self.reward_function(0, curr_angle_to_goal, True)

        # Penalty for z velocity
        # height_reward = -abs(current_base_velocity[2]**2)
        height_reward = self.penalty_function(0, current_base_velocity[2], False)
        if self._env_step_counter < 60:
            height_reward = height_reward / 10
        height_reward = height_reward / 2
        if self._objective_weights[1] == 0 or self._env_step_counter == 0:
            height_reward = 0
        elif height_reward < -5 / self._objective_weights[1]:
            height_reward = -5 / self._objective_weights[1]
        # print("current_base_velocity[2]:", current_base_velocity[2])
        # print("height_reward:", height_reward * 5 / 300)
        # Penalty for orientation velocity
        # shakevel_reward = -(abs(current_base_angvelocity[0])**2 + abs(current_base_angvelocity[1])**2)
        shakevel_reward = self.penalty_function(0, current_base_angvelocity[0], True) + self.penalty_function(0, current_base_angvelocity[1], True)
        if self._env_step_counter < 60:
            shakevel_reward = shakevel_reward / 2
        shakevel_reward = shakevel_reward / 2
        if self._objective_weights[2] == 0 or self._env_step_counter == 0:
            shakevel_reward = 0
        elif shakevel_reward < -5 / self._objective_weights[2]:
            shakevel_reward = -5 / self._objective_weights[2]
        # print("shakevel_reward:", shakevel_reward)

        # Penalty for energy consumption.
        # delta_ang = np.abs(current_joint_angles - self._last_motor_angle)
        # if self._env_step_counter < 10:
        #     # energy_reward = -np.abs(
        #     # np.dot(current_joint_torques,
        #     #        current_joint_angvelocities)) * self._time_step / 10
        #     energy_reward = -np.abs(
        #     np.dot(current_joint_torques,
        #            delta_ang)) * self._time_step / 10
        # else:
        #     # energy_reward = -np.abs(
        #     # np.dot(current_joint_torques,
        #     #        current_joint_angvelocities)) * self._time_step
        #     energy_reward = -np.abs(
        #     np.dot(current_joint_torques,
        #              delta_ang)) * self._time_step
        energy_reward = 0
        if self._env_step_counter < 10:
            for tau,vel in zip(self._dt_motor_torques,self._dt_motor_velocities):
                energy_reward += np.abs(np.dot(tau,vel)) * self._time_step / 100
        else:
            for tau,vel in zip(self._dt_motor_torques,self._dt_motor_velocities):
                energy_reward += np.abs(np.dot(tau,vel)) * self._time_step / 10
        if self._objective_weights[3] == 0:
            energy_reward = 0
        elif energy_reward < -5 / self._objective_weights[3]:
            energy_reward = -5 / self._objective_weights[3]
        
        # pelnaty for action rate
        # action_rate_reward = -np.abs(np.dot(self._delta_action, self._delta_action))
        action_rate_reward = -np.abs(np.dot(delta_action, delta_action))
        if self._env_step_counter == 0:
            action_rate_reward = 0
        if self._objective_weights[4] == 0 or self._env_step_counter == 0:
            action_rate_reward = 0
        elif action_rate_reward < -5 / self._objective_weights[4]:
            action_rate_reward = -5 / self._objective_weights[4]
        # print("action_rate_reward:", action_rate_reward)
        
        # yaw angle penalty
        _, _, yaw = quaternion_to_euler(current_orientation)
        aim_yaw = np.arctan2(current_goal_state[1], current_goal_state[0])
        yaw_penalty = self.penalty_function(aim_yaw, yaw, True)
        if self._env_step_counter == 0:
            yaw_penalty = 0
        if self._objective_weights[5] == 0:
            yaw_penalty = 0
        elif yaw_penalty < -5 / self._objective_weights[5]:
            yaw_penalty = -5 / self._objective_weights[5]

        objectives = [dist_reward, height_reward, shakevel_reward, energy_reward, action_rate_reward, yaw_penalty]
        weighted_objectives = [o * w for o, w in zip(objectives, self._objective_weights)]

        reward = sum(weighted_objectives)
        # if self.is_fallen():
        #     reward = -30*94*5
        # reward = reward / 94 / 5

        # self._objectives.append(objectives)
        # print("reward:", reward)
        return reward
    
    def get_objectives(self):
        return self._objectives
    
    @property
    def objective_weights(self):
        return self._objective_weights
    
    # 获取observation
    def _get_observation(self):
        observation = []
        observation.extend((tuple)(elem / 10 for elem in self._goal_state)) # 3
        observation.extend((tuple)(elem / 4.0 for elem in self._CPG_obs[0, :])) # 4-7
        observation.extend((self.phantomx.GetBaseOrientation())) # 8-11
        observation.extend((tuple)(elem / 4.0 for elem in (self.phantomx.GetTrueBodyLinearVelocity())))# 2 # 12-14
        observation.extend((tuple)(elem / 30.0 for elem in (self.phantomx.GetTrueBodyAngularVelocity())))# 7 # 15-17
        observation.extend((tuple)(elem / 5.0 for elem in (self.phantomx.GetTrueMotorAngles()))) # 18-35
        observation.extend((tuple)(elem / 10 for elem in (self.phantomx.GetBasePosition()))) # 36-38
        observation.extend((tuple)(elem / 1 for elem in self._last_action)) # action rate # 39-50

        # observation = observation - np.mean(observation)  
        # observation = observation / np.max(np.abs(observation)) 
        self._observation = observation
        # 观测observation是否超出上下界，并将超出的部分保存到文件中，追加写入
        if np.any(observation > self._get_observation_upper_bound()):
            with open("observation_upper_bound.txt", "a") as f:
                f.write(str(observation) + "\n")
        if np.any(observation < self._get_observation_lower_bound()):
            with open("observation_lower_bound.txt", "a") as f:
                f.write(str(observation) + "\n")

        return self._observation
    

    def _transform_action_to_motor_command(self, action):
        return action
    
    def _transform_motor_to_motor_command(self, motorcommand, action):
        motorcommand = self.phantomx.ConvertActionToLegAngle_Tripod(motorcommand, action)
        return motorcommand


    def _termination(self):
        # return self.is_fallen() or self._env_step_counter > 298
        return self.is_fallen() or self._env_step_counter >= 94*5 - 1
    
    def is_fallen(self):
        """Decide whether the phantomx has fallen.

        If the up directions between the base and the world is larger (the dot
        product is smaller than 0.85) or the base is very low on the ground
        (the height is smaller than 0.13 meter), the phantomx is considered fallen.

        Returns:
            Boolean value that indicates whether the phantomx has fallen.
        """
        orientation = self.phantomx.GetBaseOrientation()
        rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
        local_up = rot_mat[6:]
        pos = self.phantomx.GetBasePosition()
        heigh = self.phantomx.GetBaseHigh()
        # return (np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.85 or heigh < 0.06)
        return (heigh < 0.1)

    def _get_observation_upper_bound(self):
        """Get the upper bound of the observation.
        """
        upper_bound = np.zeros(50)
        upper_bound[0:3] = 1.0 # goal state
        upper_bound[3:7] = 1.0  # CPG data
        upper_bound[7:11] = 1.0 # base orientation
        upper_bound[11:14] = 1.0 # base linear vel
        upper_bound[14:17] = 1.0 # base ang vel
        for i in range(18):
            if i%3 == 0:
                upper_bound[17+i] = 1.0
            elif i%3 == 1:
                upper_bound[17+i] = 1.0
            else:
                upper_bound[17+i] = 1.0
        upper_bound[35:38] = 1.0 # base position
        upper_bound[38:50] = 1.0 # last action
        return upper_bound
    
    def _get_observation_lower_bound(self):
        lower_bound = np.zeros(50)
        lower_bound[0:3] = -1.0
        lower_bound[3:7] = -1.0
        lower_bound[7:11] = -1.0
        lower_bound[11:14] = -1.0
        lower_bound[14:17] = -1.0
        for i in range(18):
            if i%3 == 0:
                lower_bound[17+i] = -1.0
            elif i%3 == 1:
                lower_bound[17+i] = -1.0
            else:
                lower_bound[17+i] = -1.0
        lower_bound[35:53] = -1.0
        lower_bound[35:38] = -1.0
        lower_bound[38:50] = -1.0 # last action
        return lower_bound

    def _get_observation_dimension(self):
        return len(self._get_observation())
    
    def set_goal_state(self, goal_state):
        self.test_goal_state = goal_state