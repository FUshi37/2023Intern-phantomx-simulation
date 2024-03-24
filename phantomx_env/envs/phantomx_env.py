import gym
from gym import spaces
from gym.utils import seeding

import pybullet
from pybullet_utils import bullet_client as bc
import pybullet_data

import numpy as np
from gym.utils import seeding
from phantomx_env.envs.phantomx import Phantomx
import math
import time

RENDER_HEIGHT = 360
RENDER_WIDTH = 480
NUM_MOTORS = 18

class PhantomxGymEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 100}
	
    def __init__(self, urdf_root=pybullet_data.getDataPath(), 
                 render=False,
                 distance_limit=10,
                 forward_reward_cap=float("inf"), 
                 distance_weight=14.0,
                 energy_weight=0.08,
                 drift_weight=3.0,
                 shake_weight=0.03,
                 abs_x_weight=1.0,
                 x_ori_weight=0.1,
                 y_ori_weight = 0.1,
                 abs_y_weight = 1.0,
                 z_ori_weight=0.1,
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
        self._cam_dist = 1.0
        self._cam_yaw = 0
        self._cam_pitch = -30
        self._last_frame_time = 0.0
        self.control_time_step = 0.01
        self._distance_limit = distance_limit
        self._forward_reward_cap = forward_reward_cap
        self._time_step = 0.01
        self._objective_weights = [distance_weight, energy_weight, drift_weight, shake_weight, abs_x_weight, x_ori_weight, y_ori_weight, abs_y_weight, z_ori_weight]
        self._objectives = []
        if self._is_render:
            self._pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._pybullet_client = bc.BulletClient()
        
        self._hard_reset = True
		
        self.seed()
        self.reset()

        self._action_bound = 1.0
        action_dim = NUM_MOTORS
        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high)

        # self._action_bound = 1.0
        # action_dim = 3
        # action_high = np.array([self._action_bound] * action_dim)
        # self.action_space = spaces.Box(-action_high, action_high)

        # self._action_bound = 1.0
        # action_dim = 4
        # action_high = np.array([self._action_bound] * action_dim)
        # self.action_space = spaces.Box(-action_high, action_high)

        observation_high = self._get_observation_upper_bound()
        # observation_low = -observation_high
        observation_low = self._get_observation_lower_bound()
        # print("observation_upper_bound:", observation_high)
        self.observation_space = spaces.Box(observation_low, observation_high)	 
        self._hard_reset = hard_reset

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
        
        time_spent = time.time() - self._last_frame_time
        self._last_frame_time = time.time()
        time_to_sleep = self.control_time_step - time_spent
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)

        action = self._transform_action_to_motor_command(action)
        self.phantomx.Step(action)
        reward = self._reward()
        done = self._termination()
        # print("robotpos:", self.phantomx.GetBasePosition())
        # print("is_done:", done)

        self._env_step_counter += 1

        if done:
            self.phantomx.Terminate()
        
        observation = np.array(self._get_observation()).astype(np.float32)
        info = {}
        return observation, reward, done, info

    def reset(self):
	    #重新初始化
        # print("reset===========================")
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 0)
        # print("reset client done")
	
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
        self._last_base_position = [0, 0, 0]
        # print("reset debug")
        self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
                                                     self._cam_pitch, [0, 0, 0])
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

    def _reward(self):
        current_base_position = self.phantomx.GetBasePosition()
        
        # forward_reward = current_base_position[0] - self._last_base_position[0]
        forward_reward = (current_base_position[0] - self._last_base_position[0]) / self._time_step
    # Cap the forward reward if a cap is set.
        forward_reward = min(forward_reward, self._forward_reward_cap)
    # Penalty for sideways translation.
        # drift_reward = -abs(current_base_position[1] - self._last_base_position[1])
        drift_reward = -abs((current_base_position[1] - self._last_base_position[1]) / self._time_step)
    # Penalty for sideways rotation of the body.
        orientation = self.phantomx.GetBaseOrientation()
        rot_matrix = pybullet.getMatrixFromQuaternion(orientation)
        local_up_vec = rot_matrix[6:]
        shake_reward = -abs(np.dot(np.asarray([1, 1, 0]), np.asarray(local_up_vec)))
        # energy_reward = -np.abs(
        #     np.dot(self.phantomx.GetTrueMotorTorques(),
        #            self.phantomx.GetTrueMotorVelocities())) * self._time_step
        energy_reward = -abs(0.14 - current_base_position[2]) / self._time_step
        
        height_reward = -abs(current_base_position[2] - self._last_base_position[2]) / self._time_step
        # if current_base_position[2] < 0.06:
        #     height_reward = -10000
        # height_reward = -abs(0.14 - current_base_position[2]) / self._time_step
        absx_reward = current_base_position[0]
        # objectives = [forward_reward, energy_reward, drift_reward, shake_reward]
        x_ori_reward = -(abs(orientation[0])) / self._time_step
        y_ori_reward = -(abs(orientation[1])) / self._time_step
        
        absy_reward = -abs(current_base_position[1]) / self._time_step

        z_ori_reward = -(abs(orientation[2])) / self._time_step

        objectives = [forward_reward, energy_reward, drift_reward, height_reward, absx_reward, x_ori_reward, y_ori_reward, absy_reward, z_ori_reward]
        weighted_objectives = [o * w for o, w in zip(objectives, self._objective_weights)]
        reward = sum(weighted_objectives)
        self._objectives.append(objectives)
        return reward
    
    def get_objectives(self):
        return self._objectives
    
    @property
    def objective_weights(self):
        return self._objective_weights

    def _get_observation(self):
        observation = []
        # observation.extend(list(self.phantomx.GetBasePosition()))
        # observation.extend(list(self.phantomx.GetBaseOrientation()))
        # observation.extend(list(self.phantomx.GetTrueBodyLinearVelocity()))
        # observation.extend(list(self.phantomx.GetTrueBodyAngularVelocity()))
        observation.extend(self.phantomx.GetTrueMotorAngles())
        # observation.extend(self.phantomx.GetTrueMotorVelocities())
        # observation = observation - np.mean(observation)  
        # observation = observation / np.max(np.abs(observation)) 
        self._observation = observation
        # print("observation:", self._observation)
        # print("observation len:", len(self._observation))
        return self._observation
    

    def _transform_action_to_motor_command(self, action):
        # action = self.phantomx.ConvertActionToLegAngle(action)
        action = self.phantomx.ConvertActionToLegAngle_Tripod(action)
        return action
    
    def _termination(self):
        position = self.phantomx.GetBasePosition()
        distance = math.sqrt(position[0]**2 + position[1]**2)
        return self.is_fallen() or position[2]>0.4 or distance > self._distance_limit
        # return distance > self._distance_limit

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
        # upper_bound = np.zeros(31)
        # upper_bound[0:3] = self._distance_limit  # base_position
        # upper_bound[3:7] = 1.0  # base_orientation
        # upper_bound[7:10] = 3.0 #base linear vel
        # upper_bound[10:13] = 3.0 #base angular vel
        # for i in range(18):
        #     if i%3 == 0:
        #         upper_bound[13+i] = 0.523599
        #     elif i%3 == 1:
        #         upper_bound[13+i] = 1.5708
        #     else:
        #         upper_bound[13+i] = 0.6179939
        # return upper_bound
        upper_bound = np.zeros(18)
        for i in range(18):
            if i%3 == 0:
                upper_bound[i] = 0.523599
            elif i%3 == 1:
                upper_bound[i] = 1.5708
            else:
                upper_bound[i] = 0.6179939
        # upper_bound[37:61] = 3.0 #joint vel
        # upper_bound[num_motors:2 * num_motors] = (motor.MOTOR_SPEED_LIMIT)  # Joint velocity.
        # upper_bound[2 * num_motors:3 * num_motors] = (motor.OBSERVED_TORQUE_LIMIT)  # Joint torque.
        # upper_bound[3 * num_motors:] = 1.0  # Quaternion of base orientation.
        return upper_bound
    
    def _get_observation_lower_bound(self):
        # lower_bound = np.zeros(31)
        # lower_bound[0:3] = -self._distance_limit
        # lower_bound[3:7] = -1.0
        # lower_bound[7:10] = -3.0
        # lower_bound[10:13] = -3.0
        # for i in range(18):
        #     if i%3 == 0:
        #         lower_bound[13+i] = -0.523599
        #     elif i%3 == 1:
        #         lower_bound[13+i] = -1.5708
        #     else:
        #         lower_bound[13+i] = -1.5708
        # return lower_bound
        lower_bound = np.zeros(18)
        for i in range(18):
            if i%3 == 0:
                lower_bound[i] = -0.523599
            elif i%3 == 1:
                lower_bound[i] = -1.5708
            else:
                lower_bound[i] = -1.5708
        return lower_bound

    # def _get_observation_lower_bound(self):
    #     return -self._get_observation_upper_bound()

    def _get_observation_dimension(self):
        return len(self._get_observation())