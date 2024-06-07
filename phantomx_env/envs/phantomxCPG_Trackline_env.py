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
LISTLEN = 94 # 平均速度计算窗口长度，94为一个CPG周期
FREC = 100.0 # 用于计算self._time_step，主要用于确定CPG计算的步长
REWARD_FACTOR = 0.25 # 指数奖励函数时作用于分母的因子，现在没用
ACTION_REPEATE = 10 # 每个step中仿真执行次数，仿真帧率/ACTION_REPEATE即为控制帧率

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
                 x_velocity_weight = 26.0,# 26  #x方向速度跟随奖励项系数
                 y_velocity_weight = 2.00,# 2   #y方向速度跟随奖励项系数
                 yaw_velocity_weight = 2.00,# 2 #yaw角方向速度跟随奖励项系数
                 height_weight = 5.0,# 5        #高度速度惩罚项系数
                 shakevel_weight = 5.0,# 5      #抖动速度惩罚项系数
                 energy_weight = 0.5,# 0.1     #作功惩罚项系数
                 action_rate = 7.0, # 1        #action变化率惩罚项系数
                 yaw_weight = 3.0, # 3          #yaw角度惩罚项系数
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
        self._time_step = 1.0 / FREC
        self._max_episode_steps = 1024        
        self._last_reward = -np.inf
        self._cumulative_reward = 0

        self.forward_reward_list = LimitedList(LISTLEN * ACTION_REPEATE)
        self.drift_reward_list = LimitedList(LISTLEN * ACTION_REPEATE)
        self.energy_reward_list = LimitedList(LISTLEN * ACTION_REPEATE)
        self.shake_reward_list = LimitedList(LISTLEN * ACTION_REPEATE)
        self.height_reward_list = LimitedList(LISTLEN * ACTION_REPEATE)
        self.yaw_reward_list = LimitedList(LISTLEN * ACTION_REPEATE)

        # Goal State
        self._goal_state = [0, 0, 0]#x y velocity direction and magnitude
        self.test_goal_state = [0, 0, 0]
        
        # 用于作功惩罚项计算
        self._dt_motor_torques = []
        self._dt_motor_velocities = []

        self._periodData = []
        self._periodDataIndex = 0

        # 奖励函数计算各项权重
        self._objective_weights = [x_velocity_weight, y_velocity_weight, yaw_velocity_weight, height_weight, shakevel_weight, energy_weight, action_rate, yaw_weight]
        
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

        # action_dim12维，前6维为髋关节映射系数/映射系数变化值，后6维为膝、踝关节映射系数/0映射系数变化值
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

        self.observation_space = spaces.Box(observation_low, observation_high)	 
        self._hard_reset = hard_reset

        self.PltModule = PlotModuleAssistor()
        self.CPGModule = PhantomxCPG()
        self.ActionModule = ActionModuleSelector()
        self.OnlineCPGModule = OnlinePhantomxCPG()


    def step(self, action):
        """Step forward the simulation, given the action.

        Args:
        action: A list of mapping coefficient from CPG to angles for motors.

        Returns:
          observations: 
          reward: The reward for the current state-action pair.
          done: Whether the episode has ended.
          info: A dictionary that stores diagnostic information.
        """

        # self._last_base_position = self.phantomx.GetBasePosition()
        self._last_motor_angle = self.phantomx.GetTrueMotorAngles()

        self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
                                                    self._cam_pitch, self.phantomx.GetBasePosition())
        
        # time_spent = time.time() - self._last_frame_time
        # self._last_frame_time = time.time()
        # time_to_sleep = self.control_time_step - time_spent
        # if time_to_sleep > 0:
        #     time.sleep(time_to_sleep)

        # CPG步态计算
        t = np.linspace(self._env_step_counter*self._time_step, self._env_step_counter*self._time_step+self._time_step, 2)
        
        # # 在线计算CPG的值
        # data = self.OnlineCPGModule.online_calculate(t, initial_values=self.history_data[-1, :])
        
        # 直接从文件中读取储存了一个周期CPG的值
        data = self._periodData[self._periodDataIndex]
        if self._periodDataIndex < len(self._periodData)-1:
            self._periodDataIndex += 1
        else:
            self._periodDataIndex = 0

        # 矫正data类型    
        data.reshape(1, -1)
        if (len(data)==2):
            data = data[1:].reshape(1, -1)
        self._data = data

        # 取self._data 36个数据中的前四个作为observation
        self._CPG_obs = np.array([data[0][0], data[0][1], data[0][2], data[0][3]]).reshape(1, -1)
        self.history_data = np.vstack((self.history_data, data))

        # SelectAction根据三足步态调整CPG曲线，self._motorcommands为最终的CPG曲线
        self._motorcommands = self.ActionModule.SelectAction(action_mode=10, t=t, data=data[-1, :])
        # CPG步态计算结束

        # action = self._transform_action_to_motor_command(action)

        # 将调整后的CPG曲线映射到关节空间，现在函数内逻辑为髋关节action为映射系数、膝、踝关节action为映射系数变化值
        motorcommand = self._transform_motor_to_motor_command(self._motorcommands, action)

        self._dt_motor_torques = []
        self._dt_motor_velocities = []

        # 100HZ控制帧率，储存每个step中的作功、速度相关数据用于后续reward计算
        for i in range(ACTION_REPEATE):
            self.phantomx.Step(motorcommand)
            self._dt_motor_torques.append(self.phantomx.GetTrueMotorTorques())
            self._dt_motor_velocities.append(self.phantomx.GetTrueMotorVelocities())
            self.forward_reward_list.add_element(self.phantomx.GetTrueBodyLinearVelocity()[0])
            self.drift_reward_list.add_element(self.phantomx.GetTrueBodyLinearVelocity()[1])
            self.yaw_reward_list.add_element(self.phantomx.GetTrueBodyAngularVelocity()[2])

        done = self._termination()
        # truncated = self._truncation()

        info = {}

        self.history_action = motorcommand        

        if done:
            self.phantomx.Terminate()
            # info['episode'] = {
            #     'r': self._cumulative_reward,  # 累计奖励
            #     'l': self._env_step_counter              # 总步数
            # }
            # print("done and info:", info)
        
        observation = np.array(self._get_observation()).astype(np.float32)

        reward = self._reward(observation, action)

        # self._cumulative_reward += reward

        # self.phantomx.GetCollisionWithGround()

        # action rate
        self._delta_action = np.array(action) - np.array(self._last_action)
        self._last_action = action

        # info = {}

        self._env_step_counter += 1
        
        return observation, reward, done, info
        
    def reset(self):
	    #重新初始化
        
        # 初始化跟随速度目标状态self._goal_state
        self._goal_state[0] = 0.15
        self._goal_state[1] = 0
        self._goal_state[2] = 0

        # 在test时接收外部给定的目标状态
        if self._set_goal_flag:
            self._goal_state = self.test_goal_state

        # 关节角初始化
        self.history_data = np.array([0.29616917,  1.04204406, -0.37516158, -1.01642539,  0.29616917,  1.04204406,
                    -0.37516158, -1.01642539,  0.29616917,  1.04204406, -0.37516158, -1.01642539,
                    -1.97166912,  0.36889423, -1.97166912,  0.36889423,  1.86400333, -0.74193836,
                    1.86400333, -0.74193836, -1.97166912,  0.36889423, -1.97166912,  0.36889423,
                    1.86400333, -0.74193836,  1.86400333, -0.74193836, -1.97166912,  0.36889423,
                    -1.97166912, 0.36889423,  1.86400333, -0.74193836,  1.86400333, -0.74193836]).reshape(1, -1)
        
        # 读取一个周期的CPG的值
        self._periodData = []
        with open("history_data.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                self._periodData.append(np.array(eval(line)).reshape(1, -1))
        self._periodDataIndex = 0

        # reset client
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 0)
        
        # 将上一个step中用于计算各项reward的相关变量置零
        self.forward_reward_list.clear()
        self.drift_reward_list.clear()
        self.energy_reward_list.clear()
        self.shake_reward_list.clear()
        self.height_reward_list.clear()
        self.yaw_reward_list.clear()
        # 作功相关惩罚项变量置零
        self._dt_motor_torques = []
        self._dt_motor_velocities = []
        # action rate惩罚项相关变量置零
        self._last_action = [0] * 12
        self._delta_action = [0] * 12
    
        if self._hard_reset:
            self._pybullet_client.resetSimulation()

            self._pybullet_client.setGravity(0, 0, -10)

            self._ground_id = self._pybullet_client.loadURDF("%s/plane.urdf" % self._urdf_root)

            self.phantomx = Phantomx(pybullet_client=self._pybullet_client, urdf_root=self._phantomx_urdf_root)
           
            # self._obs1 = self._pybullet_client.loadURDF("%s/ObstacleReg.urdf" % self._obs_urdf_root, [-1,0,0.12], [0.707, 0, 0., 0.707], useFixedBase=True)
            # 仿真器帧率1000HZ
            self._pybullet_client.setTimeStep(1.0 / 1000)

        self.phantomx.Reset(reload_urdf=False)

        self._last_motor_angle = self.phantomx.GetTrueMotorAngles()
        self._env_step_counter = 0
        self._last_base_position = [0, 0, 0.17]

        # self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
        #                                              self._cam_pitch, [0, 0, 0])
        self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
                                                         self._cam_pitch, self.phantomx.GetBasePosition())

        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 1)

        self._cumulative_reward = 0

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
        """calculate the reward value by unimodal linear function or exponential function or flat topped linear function.

        Args:
        action: desired value and current value, yaw_flag is used to deal with calculation of yaw.

        Returns:
          value of current state.
        """
        # # 指数函数奖励函数
        # if yaw_flag:
        #     return np.e**(-abs(desired_x - current_x) / 1)
        # return np.e**(-abs(desired_x - current_x) / REWARD_FACTOR)


        # # 单峰线性函数奖励函数
        if yaw_flag:
            if current_x == desired_x:
                return 1
            elif current_x < desired_x:
                return 1 * (current_x - desired_x) + 1
            else:
                return 1 * (-current_x + desired_x) + 1
        if current_x == desired_x:
            return 1
        elif current_x < desired_x:
            return 7.5 * (current_x - desired_x) + 1
        else:
            return 7.5 * (-current_x + desired_x) + 1


        # # Action repeat -> 10 平顶线性函数
        # if yaw_flag:
        #     if current_x == desired_x:
        #         return 1
        #     elif current_x < desired_x:
        #         return 0.1 * (current_x - desired_x) + 1
        #     else:
        #         return 0.1 * (-current_x + desired_x) + 1
        # if abs(current_x - desired_x) <= self.radio_function(desired_x) or (self._env_step_counter < 94 and current_x > desired_x):
        #     # return 0.1 * (current_x - desired_x) + 1
        #     return 1
        # elif current_x < desired_x - self.radio_function(desired_x):
        #     return 2 * (current_x - desired_x) + 1
        # else:
        #     return 2 * (-current_x + desired_x) + 1
    

    def penalty_function(self, desired_x, current_x, angvel_flag):
        # 一范数计算惩罚
        if angvel_flag:
            return -0.1 * abs(current_x - desired_x)
        return -2.5 * abs(current_x - desired_x)


    def radio_function(self, desired_x):
        # if abs(desired_x) < 0.05:
        #     return 0.05
        return desired_x / 3 * 2


    def _reward(self, observations, action):
        current_goal_state = observations[0:3] * 1
        current_CPG_data = observations[3:7] * 4
        current_orientation = observations[7:11] * 1
        current_base_velocity = observations[11:14] * 4
        current_base_angvelocity = observations[14:17] * 30
        current_joint_angles = observations[17:35] * 5.0
        # current_joint_angvelocities = observations[35:53] * 20
        current_joint_angvelocities = self.phantomx.GetTrueMotorVelocities()
        # current_joint_torques = observations[53:71] * 30
        current_joint_torques = self.phantomx.GetTrueMotorTorques()
        # last_action = observations[71:83]
        # last_action = observations[53:65]
        last_action = observations[35:47]
        delta_action = action - last_action

        # velocity track reward
        x_desiredPvelocity = current_goal_state[0]
        y_desired_velocity = current_goal_state[1]
        yaw_desired_velocity = current_goal_state[2]
        # print("current_goal_state:", current_goal_state)
        


        # x velocity (froward negative x corrdinate)
        if self._env_step_counter == 0:
            # self.forward_reward_list.add_element(0)
            x_average_velocity = self.forward_reward_list.calculate_sum() / ACTION_REPEATE
        elif self._env_step_counter < LISTLEN:
            # self.forward_reward_list.add_element(current_base_velocity[0])
            x_average_velocity = self.forward_reward_list.calculate_sum() / self._env_step_counter / ACTION_REPEATE
        else:
            # self.forward_reward_list.add_element(current_base_velocity[0])
            x_average_velocity = self.forward_reward_list.calculate_sum() / LISTLEN / ACTION_REPEATE
            x_average_velocity = min(x_average_velocity, self._forward_reward_cap)

        x_velocity_reward = self.reward_function(x_desiredPvelocity, x_average_velocity, False)
        # reward限幅到 > -5
        if self._objective_weights[0] == 0:
            x_velocity_reward = 0
        # elif x_velocity_reward < -5 / self._objective_weights[0]:
        #     x_velocity_reward = -5 / self._objective_weights[0]
        self._last_average_velocity[0] = x_average_velocity
        self._last_velocity[0] = current_base_velocity[0]

        # # DEBUG
        # print("x_average_velocity:", x_average_velocity)
        # print("x_velocity", current_base_velocity[0])
        # print("x_velocity_reward:", x_velocity_reward)

        # # 将x_average_velocity保存到文件中，保存后可以通过运行test.py可视化平均速度曲线
        # with open("x_velocity.txt", "a") as f:
        #     f.write(str(x_average_velocity) + "\n")



        # y velocity (drift velocity)
        if self._env_step_counter == 0:
            # self.drift_reward_list.add_element(0)
            y_average_velocity = self.drift_reward_list.calculate_sum() / ACTION_REPEATE
        elif self._env_step_counter < LISTLEN:
            # self.drift_reward_list.add_element(current_base_velocity[1])
            y_average_velocity = self.drift_reward_list.calculate_sum() / self._env_step_counter / ACTION_REPEATE
        else:
            # self.drift_reward_list.add_element(current_base_velocity[1])
            y_average_velocity = self.drift_reward_list.calculate_sum() / LISTLEN / ACTION_REPEATE
            y_average_velocity = min(y_average_velocity, self._forward_reward_cap)
        y_velocity_reward = self.reward_function(y_desired_velocity, y_average_velocity, False)
        # reward限幅到 > -5
        if self._objective_weights[1] == 0:
            y_velocity_reward = 0
        # elif y_velocity_reward < -5 / self._objective_weights[1]:
        #     y_velocity_reward = -5 / self._objective_weights[1]
        # print("y_average_velocity:", y_average_velocity)
            


        # yaw velocity (shake velocity)
        if self._env_step_counter == 0:
            # self.yaw_reward_list.add_element(0)
            yaw_average_velocity = self.yaw_reward_list.calculate_sum() / ACTION_REPEATE
        elif self._env_step_counter < LISTLEN:
            # self.yaw_reward_list.add_element(current_base_angvelocity[2])
            yaw_average_velocity = self.yaw_reward_list.calculate_sum() / self._env_step_counter / ACTION_REPEATE
        else:
            # self.yaw_reward_list.add_element(current_base_angvelocity[2])
            yaw_average_velocity = self.yaw_reward_list.calculate_sum() / LISTLEN / ACTION_REPEATE
            yaw_average_velocity = min(yaw_average_velocity, self._forward_reward_cap)
        yaw_velocity_reward = self.reward_function(yaw_desired_velocity, yaw_average_velocity, True)
        # reward限幅到 > -5
        if self._objective_weights[2] == 0:
            yaw_velocity_reward = 0
        # elif yaw_velocity_reward < -5 / self._objective_weights[2]:
        #     yaw_velocity_reward = -5 / self._objective_weights[2]
        
        # # DEBUG
        # print("yaw_average_velocity:", yaw_average_velocity)
        # yaw_velocity_reward = self.reward_function(yaw_desired_velocity, current_base_angvelocity[2], True)



        # Penalty for z velocity
        # height_reward = -abs(current_base_velocity[2]**2)
        height_reward = self.penalty_function(0, current_base_velocity[2], False)
        if self._env_step_counter < 60:
            height_reward = height_reward / 10
        height_reward = height_reward / 2
        # reward限幅到 > -5
        if self._objective_weights[3] == 0 or self._env_step_counter == 0:
            height_reward = 0
        # elif height_reward < -5 / self._objective_weights[3]:
        #     height_reward = -5 / self._objective_weights[3]
        
        # # DEBUG
        # print("current_base_velocity[2]:", current_base_velocity[2])
        # print("height_reward:", height_reward * 5 / 300)
        
        
        
        
        # Penalty for orientation velocity
        # shakevel_reward = -(abs(current_base_angvelocity[0])**2 + abs(current_base_angvelocity[1])**2)
        shakevel_reward = self.penalty_function(0, current_base_angvelocity[0], True) + self.penalty_function(0, current_base_angvelocity[1], True)
        if self._env_step_counter < 60:
            shakevel_reward = shakevel_reward / 2
        shakevel_reward = shakevel_reward / 2
        # reward限幅到 > -5
        if self._objective_weights[4] == 0 or self._env_step_counter == 0:
            shakevel_reward = 0
        # elif shakevel_reward < -5 / self._objective_weights[4]:
        #     shakevel_reward = -5 / self._objective_weights[4]
        # # DEBUG
        # print("shakevel_reward:", shakevel_reward)



        # Penalty for energy consumption.
        energy_reward = 0
        if self._env_step_counter < 10:
            for tau,vel in zip(self._dt_motor_torques,self._dt_motor_velocities):
                energy_reward += -np.abs(np.dot(tau,vel)) * self._time_step / 100
        else:
            for tau,vel in zip(self._dt_motor_torques,self._dt_motor_velocities):
                energy_reward += -np.abs(np.dot(tau,vel)) * self._time_step / 10
        # reward限幅到 > -5
        if self._objective_weights[5] == 0:
            energy_reward = 0
        # elif energy_reward < -5 / self._objective_weights[5]:
        #     energy_reward = -5 / self._objective_weights[5]



        # pelnaty for action rate
        action_rate_reward = -np.abs(np.dot(delta_action, delta_action))
        if self._env_step_counter == 0:
            action_rate_reward = 0
        # reward限幅到 > -5
        if self._objective_weights[6] == 0 or self._env_step_counter == 0:
            action_rate_reward = 0
        # elif action_rate_reward < -5 / self._objective_weights[6]:
        #     action_rate_reward = -5 / self._objective_weights[6]
        # print("action_rate_reward:", action_rate_reward)
        


        # yaw angle penalty
        _, _, yaw = quaternion_to_euler(current_orientation)
        aim_yaw = np.arctan2(current_goal_state[1], current_goal_state[0])
        yaw_penalty = self.penalty_function(aim_yaw, yaw, False)
        if self._env_step_counter == 0:
            yaw_penalty = 0
        # reward限幅到 > -5
        if self._objective_weights[7] == 0:
            yaw_penalty = 0
        # elif yaw_penalty < -5 / self._objective_weights[7]:
        #     yaw_penalty = -5 / self._objective_weights[7]


        objectives = [x_velocity_reward, y_velocity_reward, yaw_velocity_reward, height_reward, shakevel_reward, energy_reward, action_rate_reward, yaw_penalty]        
        weighted_objectives = [o * w for o, w in zip(objectives, self._objective_weights)]
        
        # # DEBUG
        # print("x_velocity_reward:", weighted_objectives[0])
        # print("y_velocity_reward:", weighted_objectives[1])
        # print("yaw_velocity_reward:", weighted_objectives[2])
        # print("height_reward:", weighted_objectives[3])
        # print("shakevel_reward:", weighted_objectives[4])
        # print("energy_reward:", weighted_objectives[5])
        # print("action_rate_reward:", weighted_objectives[6])
        # print("yaw_penalty:", weighted_objectives[7])


        reward = sum(weighted_objectives)
        if self.is_fallen():
            reward = -30*94*5

        # normalize reward
        reward = reward / 94 / 5

        # reward = max(reward, 0)
        # self._objectives.append(objectives)
        # print("reward:", reward)

        return reward
    
    
    @property
    def objective_weights(self):
        return self._objective_weights
    
    # 获取observation
    def _get_observation(self):
        observation = []
        observation.extend((tuple)(elem / 1 for elem in self._goal_state)) # 3 # 目标速度状态
        # observation.extend((tuple)(elem / 4.0 for elem in self._data[0, :]))
        observation.extend((tuple)(elem / 4.0 for elem in self._CPG_obs[0, :])) # 4-7 # CPG数据
        observation.extend((self.phantomx.GetBaseOrientation())) # 8-11 # 机器人基体姿态
        observation.extend((tuple)(elem / 4.0 for elem in (self.phantomx.GetTrueBodyLinearVelocity())))# 2 # 12-14 # 机器人基体线速度
        observation.extend((tuple)(elem / 30.0 for elem in (self.phantomx.GetTrueBodyAngularVelocity())))# 7 # 15-17 # 机器人基体角速度
        observation.extend((tuple)(elem / 5.0 for elem in (self.phantomx.GetTrueMotorAngles()))) # 18-35 # 关节角度
        # observation.extend((tuple)(elem / 20.0 for elem in self.))# 18-35
        # observation.extend((tuple)(elem / 20.0 for elem in self.phantomx.GetTrueMotorVelocities()))
        # observation.extend((tuple)(elem / 30.0 for elem in self.phantomx.GetTrueMotorTorques()))# 30
        observation.extend((tuple)(elem / 1 for elem in self._last_action)) # action rate # 36-47 # 上一步action
        observation.extend(self.phantomx.GetCollisionWithGround())

        # observation.extend((tuple)(elem / 4 for elem in self._data[0, :]))
        # print("observation:", observation)
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

        return self.is_fallen() or self._env_step_counter >= 94*5 - 1
    
    def _truncation(self):
        return self._env_step_counter >= 94*5 - 1
    
    def is_fallen(self):
        """Decide whether the phantomx has fallen.

        If the robot's height lower than 0.1
        Returns:
            Boolean value that indicates whether the robot has fallen.
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
        # upper_bound = np.zeros(71)
        # upper_bound = np.zeros(83)
        # upper_bound = np.zeros(77)
        # upper_bound = np.zeros(65)
        # upper_bound = np.zeros(47)
        upper_bound = np.zeros(53)
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
        # upper_bound[35:53] = 1.0 #joint vel
        # upper_bound[53:71] = 1.0 #joint torque
        # # upper_bound[71:77] = 1.0
        # upper_bound[71:83] = 1.0 # last action
        # upper_bound[53:65] = 1.0 # last action
        upper_bound[35:47] = 1.0 # last action
        upper_bound[47:53] = 1.0 # collision
        return upper_bound
    
    def _get_observation_lower_bound(self):
        """Get the lower bound of the observation.
        """
        # lower_bound = np.zeros(71)
        # lower_bound = np.zeros(83)
        # lower_bound = np.zeros(77)
        # lower_bound = np.zeros(65)
        # lower_bound = np.zeros(47)
        lower_bound = np.zeros(53)
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
        # lower_bound[53:71] = -1.0
        # # lower_bound[71:77] = -1.0
        # lower_bound[71:83] = -1.0
        # lower_bound[53:65] = -1.0 # last action
        lower_bound[35:47] = -1.0 # last action
        lower_bound[47:53] = -1.0
        return lower_bound

    def _get_observation_dimension(self):
        return len(self._get_observation())
    
    def set_goal_state(self, goal_state):
        self.test_goal_state = goal_state