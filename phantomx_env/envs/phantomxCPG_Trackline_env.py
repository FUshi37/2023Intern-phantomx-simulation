import gym
from gym import spaces
from gym.utils import seeding

import pybullet
from pybullet_utils import bullet_client as bc
import pybullet_data

import numpy as np
import random
from gym.utils import seeding
from phantomx_env.envs.phantomxCPG import Phantomx
import math
import time

import sys
sys.path.append('/home/yangzhe/Intern/simulation/RL_phantomx_pybullet/src')

from src.CentralPatternGenerators.Hpof import PhantomxCPG
from src.AssistModulesCode.MatPlotAssitor import PlotModuleAssistor
from src.AssistModulesCode.ActionSelector import ActionModuleSelector
from src.CentralPatternGenerators.OnlineCPG import OnlinePhantomxCPG

RENDER_HEIGHT = 360
RENDER_WIDTH = 480
NUM_MOTORS = 18
LISTLEN = 500

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
                 distance_limit=3,
                 forward_reward_cap=float("inf"), 
                 distance_weight = 50.0,# 50
                 energy_weight = 0.5,#0.5
                 drift_weight = 20.0,#20
                 shake_weight = 20.0,#20
                 height_weight = 20.0,#20
                 shakevel_weight = 2.0,#2
                 hard_reset=True,
                 phantomx_urdf_root="/home/yangzhe/Intern/simulation/RL_phantomx_pybullet/phantomx_description"):
                # phantomx_urdf_root="/home/yangzhe/Intern/simulation/RL_phantomx_pybullet/hexapod_34/urdf"):
        super(PhantomxGymEnv, self).__init__()  
        self._urdf_root = urdf_root
        self._phantomx_urdf_root = phantomx_urdf_root
        self._obs_urdf_root = "/home/yangzhe/Intern/simulation/RL_phantomx_pybullet/ObstacleReg/urdf"
        self._observation = []
        self._norm_observation = []
        self._env_step_counter = 0
        self._is_render = render
        self._cam_dist = 2.0
        self._cam_yaw = 0
        self._cam_pitch = -30
        self._last_frame_time = 0.0
        self.control_time_step = 0.01
        self._distance_limit = distance_limit
        self._forward_reward_cap = forward_reward_cap
        self._action_change_cap = 0
        self._time_step = 0.01
        self._max_episode_steps = 1024
        self._velrewardlist = []
        
        self.initial_action = [-0.523599, -0.523599, -0.523599, 0.523599, 0.523599, 0.523599, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25]

        self.forward_reward = 0
        self.drift_reward = 0
        self.energy_reward = 0
        self.shake_reward = 0
        self.height_reward = 0
        
        self.forward_reward_list = LimitedList(LISTLEN)
        self.drift_reward_list = LimitedList(LISTLEN)
        self.energy_reward_list = LimitedList(LISTLEN)
        self.shake_reward_list = LimitedList(LISTLEN)
        self.height_reward_list = LimitedList(LISTLEN)

        # Goal State
        self._goal_posture = []
        self._goal_velocity = []
        self._goal_state = [0, 0, 0, 0, 0, 0]#linear velocity & orientataion velocity
        
        self._objective_weights = [distance_weight, drift_weight, energy_weight, shake_weight, height_weight, shakevel_weight]
        self._objectives = []
        if self._is_render:
            self._pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._pybullet_client = bc.BulletClient()
        
        # 机器人运动关节角初始值
        self.history_data = np.array([0.29616917,  1.04204406, -0.37516158, -1.01642539,  0.29616917,  1.04204406,
                    -0.37516158, -1.01642539,  0.29616917,  1.04204406, -0.37516158, -1.01642539,
                    -1.97166912,  0.36889423, -1.97166912,  0.36889423,  1.86400333, -0.74193836,
                    1.86400333, -0.74193836, -1.97166912,  0.36889423, -1.97166912,  0.36889423,
                    1.86400333, -0.74193836,  1.86400333, -0.74193836, -1.97166912,  0.36889423,
                    -1.97166912, 0.36889423,  1.86400333, -0.74193836,  1.86400333, -0.74193836]).reshape(1, -1)
        self._data = self.history_data.reshape(1, -1)
        self._hard_reset = True
		
        self.seed()
        self.reset()

        # self._action_bound = 3.0
        self._action_bound = 1.0
        # action_dim = NUM_MOTORS
        action_dim = 12
        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high)

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
        self._last_base_position = self.phantomx.GetBasePosition()

        # print("action: ", action)
        self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
                                                    self._cam_pitch, self.phantomx.GetBasePosition())
        
        time_spent = time.time() - self._last_frame_time
        self._last_frame_time = time.time()
        time_to_sleep = self.control_time_step - time_spent
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)

        # CPG步态计算
        t = np.linspace(self._env_step_counter*0.01, self._env_step_counter*0.01+0.01, 2)
        data = self.OnlineCPGModule.online_calculate(t, initial_values=self.history_data[-1, :])
        if (len(data)==2):
            data = data[1:].reshape(1, -1)
        self._data = data
        self.history_data = np.vstack((self.history_data, data))

        self._motorcommands = self.ActionModule.SelectAction(action_mode=10, t=t, data=data[-1, :])
        # print("DATA:", data)
        # print("HISTORY_DATA:", self.history_data)
        # print("MOTORCOMMANDS:", self._motorcommands)
        # CPG步态计算结束

        # if self._env_step_counter < 100:
        #     action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # else :
        #     action = self._transform_action_to_motor_command(action)
        action = self._transform_action_to_motor_command(action)
        # print("motorcommand:", self._motorcommands)
        # print("action: ", action)
        motorcommand = self._transform_motor_to_motor_command(self._motorcommands, action)
        # action = self._transform_action_to_motor_command(action)
        self.phantomx.Step(motorcommand)
        # print reward
        # print("reward:", reward)

        done = self._termination()
        # print("robotpos:", self.phantomx.GetBasePosition())
        # if done:
        #     print("is_done:", done)

        self.history_action = motorcommand

        self._env_step_counter += 1
        

        if done:
            self.phantomx.Terminate()
        
        observation = np.array(self._get_observation()).astype(np.float32)
        reward = self._reward(observation)
        info = {}
        return observation, reward, done, info
        
    def reset(self):
	    #重新初始化
        for i in range(6):
            if (i == 0 or i == 1 or i == 5):
                self._goal_state[i] = random.uniform(0, 5)
        # print("goal_state: ", self._goal_state)
        #关节角初始化
        self.history_data = np.array([0.29616917,  1.04204406, -0.37516158, -1.01642539,  0.29616917,  1.04204406,
                    -0.37516158, -1.01642539,  0.29616917,  1.04204406, -0.37516158, -1.01642539,
                    -1.97166912,  0.36889423, -1.97166912,  0.36889423,  1.86400333, -0.74193836,
                    1.86400333, -0.74193836, -1.97166912,  0.36889423, -1.97166912,  0.36889423,
                    1.86400333, -0.74193836,  1.86400333, -0.74193836, -1.97166912,  0.36889423,
                    -1.97166912, 0.36889423,  1.86400333, -0.74193836,  1.86400333, -0.74193836]).reshape(1, -1)
        # print("reset===========================")
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 0)
        # print("reset client done")
        self.forward_reward_list.clear()
        self.drift_reward_list.clear()
        self.energy_reward_list.clear()
        self.shake_reward_list.clear()
        self.height_reward_list.clear()
    
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
        
        self.phantomx.Reset(reload_urdf=False)
        # print("reset done")
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

    # 只有x方向平均速度
    def _reward(self, observations):
        current_goal_sate = observations[0:6] * 5     
        current_CPG_data = observations[6:42]
        current_orientation = observations[42:46]
        current_base_velocity = observations[46:49]
        current_base_angvelocity = observations[49:52]
        current_joint_angles = observations[52:70]
        current_joint_angvelocities = observations[70:88]
        current_joint_torques = observations[88:106]

        # velocity track reward
        x_velocity = current_base_velocity[0]
        y_velocity = current_base_velocity[1]
        yaw_velocity = current_base_angvelocity[2]
        

        # forward velocity (froward negative x corrdinate)
        if self._env_step_counter == 0:
            self.forward_reward_list.add_element(0)
            forward_reward = self.forward_reward_list.calculate_sum()
        elif self._env_step_counter < LISTLEN:
            self.forward_reward_list.add_element(-current_base_velocity[0])
            forward_reward = self.forward_reward_list.calculate_sum() / self._env_step_counter
        else:
            self.forward_reward_list.add_element(-current_base_velocity[0])
            forward_reward = self.forward_reward_list.calculate_sum() / LISTLEN
            forward_reward = min(forward_reward, self._forward_reward_cap)

        # Cap the forward reward if a cap is set.
        self.forward_reward = min(self.forward_reward, self._forward_reward_cap)
        
        # Penalty for sideways translation.
        # drift velocity
        self.drift_reward = -abs(current_basey_position)
        drift_reward = self.drift_reward

        # Penalty for sideways rotation of the body.
        shake_reward = -abs(abs(orientation[0]) + abs(orientation[1]) + abs(orientation[2]))

        # Penalty for energy consumption.
        if self._env_step_counter < 30:
            energy_reward = -np.abs(
            np.dot(current_joint_torques,
                   current_joint_angvelocities)) * self._time_step / 10
        else:
            energy_reward = -np.abs(
            np.dot(current_joint_torques,
                   current_joint_angvelocities)) * self._time_step

        # Penalty for height of the body.
        height_reward = -abs(current_base_position[2] - self._last_base_position[2]) / self._time_step
        height_reward = self.height_reward
        
        # Penalty for orientation velocity
        shakevel_reward = -(abs(current_base_angvelocity[0]) + abs(current_base_angvelocity[1]) + abs(current_base_angvelocity[2]))
        if self._env_step_counter == 0:
            shakevel_reward = 0

        objectives = [forward_reward, drift_reward, energy_reward, shake_reward, height_reward, shakevel_reward]
        
        weighted_objectives = [o * w for o, w in zip(objectives, self._objective_weights)]
        reward = sum(weighted_objectives)
        self._objectives.append(objectives)
        # print("reward:", reward)
        return reward
    
    def get_objectives(self):
        return self._objectives
    
    @property
    def objective_weights(self):
        return self._objective_weights

    # # 沿着x轴负方向走需要的observation
    # def _get_observation(self):
    #     observation = []
    #     position_list = []
    #     position_list.append(self.phantomx.GetBasePosition()[1]/10.0)
    #     # observation.extend(list(self.phantomx.GetBasePosition()[1])/10)
    #     observation.extend(position_list)
    #     # observation.extend((self.phantomx.GetBasePosition()))
    #     observation.extend((self.phantomx.GetBaseOrientation()))#/1
    #     # observation.extend((self.phantomx.GetTrueBodyLinearVelocity()))#/5
    #     # observation.extend((self.phantomx.GetTrueBodyAngularVelocity()))#/10
    #     # observation.extend((self.phantomx.GetTrueMotorAngles())/1.5)#/
    #     # observation.extend(self.phantomx.GetTrueMotorVelocities()/20.0)#/20
    #     observation.extend((tuple)(elem / 5.0 for elem in (self.phantomx.GetTrueBodyLinearVelocity())))
    #     observation.extend((tuple)(elem /10.0 for elem in (self.phantomx.GetTrueBodyAngularVelocity())))
    #     observation.extend((tuple)(elem /1.5 for elem in (self.phantomx.GetTrueMotorAngles())))
    #     observation.extend((tuple)(elem / 20.0 for elem in self.phantomx.GetTrueMotorVelocities()))
    #     observation.extend((tuple)(elem / 50.0 for elem in self.phantomx.GetTrueMotorTorques()))
    #     observation.extend((tuple)(elem / 4 for elem in self._data[0, :]))
    #     # print("observation:", observation)
    #     # observation = observation - np.mean(observation)  
    #     # observation = observation / np.max(np.abs(observation)) 
    #     self._observation = observation

    #     return self._observation
    
    # 获取observation
    def _get_observation(self):
        observation = []
        observation.extend((tuple)(elem / 5.0 for elem in self._goal_state))
        observation.extend((tuple)(elem / 4.0 for elem in self._data[0, :]))
        observation.extend((self.phantomx.GetBaseOrientation()))
        observation.extend((tuple)(elem / 5.0 for elem in (self.phantomx.GetTrueBodyLinearVelocity())))
        observation.extend((tuple)(elem / 10.0 for elem in (self.phantomx.GetTrueBodyAngularVelocity())))
        observation.extend((tuple)(elem / 1.5 for elem in (self.phantomx.GetTrueMotorAngles())))
        observation.extend((tuple)(elem / 20.0 for elem in self.phantomx.GetTrueMotorVelocities()))
        observation.extend((tuple)(elem / 50.0 for elem in self.phantomx.GetTrueMotorTorques()))
        # observation.extend((tuple)(elem / 5.0 for elem in (self.phantomx.GetTrueBodyLinearVelocity())))
        # observation.extend((tuple)(elem /10.0 for elem in (self.phantomx.GetTrueBodyAngularVelocity())))
        # observation.extend((tuple)(elem /1.5 for elem in (self.phantomx.GetTrueMotorAngles())))
        # observation.extend((tuple)(elem / 20.0 for elem in self.phantomx.GetTrueMotorVelocities()))
        # observation.extend((tuple)(elem / 50.0 for elem in self.phantomx.GetTrueMotorTorques()))
        # observation.extend((tuple)(elem / 4 for elem in self._data[0, :]))
        # print("observation:", observation)
        # observation = observation - np.mean(observation)  
        # observation = observation / np.max(np.abs(observation)) 
        self._observation = observation

        return self._observation
    

    def _transform_action_to_motor_command(self, action):
        return action
    
    def _transform_motor_to_motor_command(self, motorcommand, action):
        motorcommand = self.phantomx.ConvertActionToLegAngle_Tripod(motorcommand, action)
        return motorcommand


    def _termination(self):

        return self.is_fallen() or self._env_step_counter > 2048
    
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
        return (heigh < 0.06)

    def _get_observation_upper_bound(self):
        """Get the upper bound of the observation.
        """
        upper_bound = np.zeros(106)
        upper_bound[0:6] = 1.0 # goal state
        upper_bound[6:42] = 1.0  # CPG data
        upper_bound[42:46] = 1.0 # base orientation
        upper_bound[46:49] = 1.0 # base linear vel
        upper_bound[49:52] = 1.0 # base ang vel
        for i in range(18):
            if i%3 == 0:
                upper_bound[52+i] = 1.0#0.7
            elif i%3 == 1:
                upper_bound[52+i] = 1.0#1.5
            else:
                upper_bound[52+i] = 1.0#1.5
        upper_bound[70:88] = 1.0 #joint vel
        upper_bound[88:106] = 1.0 #joint torque

        return upper_bound
    
    def _get_observation_lower_bound(self):

        lower_bound = np.zeros(106)
        lower_bound[0:6] = -1.0 # goal state
        lower_bound[6:42] = -1.0  # CPG data
        lower_bound[42:46] = -1.0 # base orientation
        lower_bound[46:49] = -1.0 # base linear vel
        lower_bound[49:52] = -1.0 # base ang vel
        for i in range(18):
            if i%3 == 0:
                lower_bound[52+i] = -1.0#0.7
            elif i%3 == 1:
                lower_bound[52+i] = -1.0#1.5
            else:
                lower_bound[52+i] = -1.0#1.5
        lower_bound[70:88] = -1.0 #joint vel
        lower_bound[88:106] = -1.0 #joint torque

        return lower_bound

    def _get_observation_dimension(self):
        return len(self._get_observation())