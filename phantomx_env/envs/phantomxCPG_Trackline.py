import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import copy
import math
import numpy as np

INIT_POSITION = [0, 0, 0.17]
INIT_ORIENTATION = [0, 0, 0, 1]
LEG_POSITION = ["leg1", "leg2", "leg3", "leg4", "leg5", "leg6"]
MOTOR_NAMES = [
    "j_c1_rf", "j_thigh_rf", "j_tibia_rf",
    "j_c1_rm", "j_thigh_rm", "j_tibia_rm",
    "j_c1_rr", "j_thigh_rr", "j_tibia_rr",
    "j_c1_lf", "j_thigh_lf", "j_tibia_lf",
    "j_c1_lm", "j_thigh_lm", "j_tibia_lm",
    "j_c1_lr", "j_thigh_lr", "j_tibia_lr",
]
LINK_NAMES = [
    "c1_rf", "thigh_rf", "tibia_rf",
    "c1_rm", "thigh_rm", "tibia_rm",
    "c1_rr", "thigh_rr", "tibia_rr",
    "c1_lf", "thigh_lf", "tibia_lf",
    "c1_lm", "thigh_lm", "tibia_lm",
    "c1_lr", "thigh_lr", "tibia_lr",
]

A1 = 0.523599
B1 = 0
A2 = -0.25
B2 = 0.8
A3 = 0.25
B3 = 0.3

Initial_Action = [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25]


class Phantomx:
    def __init__(self,
               pybullet_client,
               urdf_root="",
               cmd_vel = [1, 0, 0],
               ):
        self.num_motors = 18
        self.num_legs = int(self.num_motors / 3)
        self._pybullet_client = pybullet_client      
        self._urdf_root = urdf_root
        self.time_step = 0.01
        self._motor_velocity_limit = 5.6548668
        #self._observation_history = collections.deque(maxlen=100)
        self._action_history = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # self._action_history = [0, 0, 0]
        # self._action_history = [0, 0.4, 0.3, 0]
        self._cmd_vel = cmd_vel
        self._step_counter = 0
        
        self.joint_leg_joint_id = [[0, 2, 3], [4, 6, 7], [8, 10, 11], [12, 14, 15], [16, 18, 19], [20, 22, 23]]
        self.joint_id = [0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23]
        
        self._joint_history_t1 = np.zeros((1,18))
        self._joint_history_t2 = np.zeros((1,18))
        self._joint_position   = np.zeros((1,18))
        self.Reset()

        # self.forward_flag = 0.0

        self.TestTargetVelocity = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def Step(self, action):
        self.ApplyAction(action)
        self._pybullet_client.stepSimulation()
        self.ReceiveObservation()
        self._step_counter += 1
 
    def Terminate(self):
        pass

    def _BuildJointNameToIdDict(self):
        """
    self._joint_name_to_id的内容：
    {'j_c1_rf': 0, 'j_thigh_rf': 2, 'j_tibia_rf': 3, 'j_c2_rf': 1,
     'j_c1_rm': 4, 'j_thigh_rm': 6, 'j_tibia_rm': 7, 'j_c2_rm': 5,
     'j_c1_rr': 8, 'j_thigh_rr': 10, 'j_tibia_rr': 11, 'j_c2_rr': 9,
     'j_c1_lf': 12, 'j_thigh_lf': 14, 'j_tibia_lf': 15, 'j_c2_lf': 13,
     'j_c1_lm': 16, 'j_thigh_lm': 18, 'j_tibia_lm': 19, 'j_c2_lm': 17,
     'j_c1_lr': 20, 'j_thigh_lr': 22, 'j_tibia_lr': 23, 'j_c2_lr': 21}
        """    
        num_joints = self._pybullet_client.getNumJoints(self.my_phantomx)
        self._joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.my_phantomx, i)
            self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]  #[0]是jointindex， [1]jointname

        # print(self._joint_name_to_id)
            
    def _BuildMotorIdList(self):
        #[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]
        self._motor_id_list = [self._joint_name_to_id[motor_name] for motor_name in MOTOR_NAMES]
        
        # print(self._motor_id_list)

    def _BuildLinkNameToIdDict(self):
        num_joints = self._pybullet_client.getNumJoints(self.my_phantomx)
        self._link_name_to_id = {}
        for i in range(num_joints):
            link_info = self._pybullet_client.getJointInfo(self.my_phantomx, i)
            self._link_name_to_id[link_info[12].decode("UTF-8")] = link_info[0]

    def GetTrueObservation(self):
        observation = []
        observation.extend(self.GetBasePosition())            #机体位置向量
        observation.extend(self.GetBaseOrientation())     #机体方向向量
        observation.extend(self.GetTrueBodyLinearVelocity())  #body线速度
        observation.extend(self.GetTrueBodyAngularVelocity())  #body角速度
        #observation.extend(self.GetBaseHigh())  #body高度
        observation.extend(self.GetTrueMotorAngles())         #关节角度
        # observation.extend(self.GetTrueMotorVelocities())     #关节速度
        #observation.extend(self.GetMotorAnglesHistoryT1())#关节历史位置状态信息t-0.01
        #observation.extend(self.GetMotorAnglesHistoryT2())#关节历史位置状态信息t-0.02
        #observation.extend(self._action_history)
        #observation.extend(self._cmd_vel)                     #速度命令
             #腿相位
        return observation

    def ReceiveObservation(self):
        """Receive the observation from sensors.

        This function is called once per step. The observations are only updated
        when this function is called.
        """
        #self._observation_history.appendleft(self.GetTrueObservation())
        self._control_observation = self.GetTrueObservation()

    def Reset(self, reload_urdf=True):
        init_position = INIT_POSITION
        if reload_urdf:
            self.my_phantomx = self._pybullet_client.loadURDF(
                "%s/phantomx.urdf" % self._urdf_root,
                init_position,
                flags=self._pybullet_client.URDF_USE_SELF_COLLISION)

            self._BuildJointNameToIdDict()
            self._BuildMotorIdList()
            # self.change_dynamics()
            self.ResetPose()

        else:
            self._pybullet_client.resetBasePositionAndOrientation(self.my_phantomx, init_position,
                                                                  INIT_ORIENTATION)
            self._pybullet_client.resetBaseVelocity(self.my_phantomx, [0, 0, 0], [0, 0, 0])   
            self.ResetPose() 
        self._step_counter = 0
        self._joint_position = [
            self._pybullet_client.getJointState(self.my_phantomx, motor_id)[0]
            for motor_id in self._motor_id_list
        ]
        self._joint_history_t1 = self._joint_position
        self._joint_history_t2 = self._joint_position
        #self._observation_history.clear()


    def ResetPose(self):
        """Reset the pose of the hexapod.
        """
        for i in range(self.num_legs):
            self._ResetPoseForLeg(i)
#   1.22501186e-04  7.99344615e-01  3.00340822e-01  1.75018622e-04
#   7.99868396e-01  3.00312262e-01  1.74642961e-04  7.99663365e-01
#   3.02052127e-01 -6.21228992e-05  7.99279627e-01  3.00390245e-01
#  -3.77734122e-05  7.99879306e-01  3.01041978e-01  2.52956109e-05
#   7.99746765e-01  3.01888409e-01
    def _ResetPoseForLeg(self, leg_id):
        """Reset the initial pose for the leg.

        Args:
            leg_id: It should be 0, 1, 2, 3, 4, 5, 6....
        """
        if leg_id % 2 == 0:
            # targetPositions = [0, -1.09956, 0.0707954249999998]
            targetPositions = [0, 0.799, 0.300]
        else:
            targetPositions = [0, 0.799, 0.300]
        self._pybullet_client.setJointMotorControlArray(self.my_phantomx,
              jointIndices=self.joint_leg_joint_id[leg_id],
              controlMode=self._pybullet_client.POSITION_CONTROL,
            #   targetPositions=[0, 0, 0],
              targetPositions = targetPositions,
              forces=[30, 30, 30],
            #   forces = [2.8, 2.8, 2.8],
              )
        # self._pybullet_client.setJointMotorControl2(
        #   bodyIndex=self.my_phantomx,
        #   jointIndex=self.joint_leg_joint_id[leg_id][0],
        #   controlMode=self._pybullet_client.POSITION_CONTROL,
        #   targetPosition=0,
        #   force=30)
        # self._pybullet_client.setJointMotorControl2(
        #   bodyIndex=self.my_phantomx,
        #   jointIndex=self.joint_leg_joint_id[leg_id][1],
        #   controlMode=self._pybullet_client.POSITION_CONTROL,
        #   targetPosition=0,
        #   force=30)
        # self._pybullet_client.setJointMotorControl2(
        #   bodyIndex=self.my_phantomx,
        #   jointIndex=self.joint_leg_joint_id[leg_id][2],
        #   controlMode=self._pybullet_client.POSITION_CONTROL,
        #   targetPosition=0,
        #   force=30)
        # self._pybullet_client.setJointMotorControl2(
        #   bodyIndex=self.my_phantomx,
        #   jointIndex=self.joint_leg_joint_id[leg_id][3],
        #   controlMode=self._pybullet_client.POSITION_CONTROL,
        #   targetPosition=0,
        #   force=30)
    
    def change_dynamics(self):
        self._pybullet_client.changeDynamics(self.my_phantomx, 3, lateralFriction=4.5, frictionAnchor=1)
        self._pybullet_client.changeDynamics(self.my_phantomx, 7, lateralFriction=4.5, frictionAnchor=1)
        self._pybullet_client.changeDynamics(self.my_phantomx, 11, lateralFriction=4.5, frictionAnchor=1)
        self._pybullet_client.changeDynamics(self.my_phantomx, 15, lateralFriction=4.5, frictionAnchor=1)
        self._pybullet_client.changeDynamics(self.my_phantomx, 19, lateralFriction=4.5, frictionAnchor=1)
        self._pybullet_client.changeDynamics(self.my_phantomx, 23, lateralFriction=4.5, frictionAnchor=1)
        pass

    def reset_action(self, actions):
        self._action_history = actions

    def map_range(self, value, from_min, from_max, to_min, to_max):
        from_span = from_max - from_min
        to_span = to_max - to_min
        value_scaled = float(value - from_min) / float(from_span)
        return to_min + (value_scaled * to_span)
    
    def ConvertActionToLegAngle(self, actions):
        """
        关节角度action∈[-1, 1]转换成实际关节角度[limit_min, limit_max]
        """
        joint_angle = copy.deepcopy(actions)
        for i in range(self.num_motors):
            if i%3 == 0:
                joint_angle[i] = self.map_range(actions[i], -1.0, 1.0, -0.523599, 0.523599)
            elif i%3 == 1:
                joint_angle[i] = self.map_range(actions[i], -0, 2.0, 0, 1.0)
            else:
                joint_angle[i] = self.map_range(actions[i], -0.0, 2.0, 0, 0.6179939)
                # joint_angle[i] = 0.1
        return joint_angle
        joint_angle[0] = self.map_range(actions[0], -1.0, 1.0, -0.523599, 0.523599)
        joint_angle[1] = self.map_range(actions[1], -1.0, 1.0, -1.5708, 1.5708)
        # joint_angle[1] = self.map_range(actions[1], -1.0, 1.0, -1.0, 1.0)
        # joint_angle[2] = self.map_range(actions[2], -1.0, 1.0, -1.5708, 0.6179939)
        joint_angle[2] = self.map_range(actions[2], -1.0, 1.0, -0.3, 0.3)
        # joint_angle[3] = actions[3]
        return joint_angle
    
    def ConvertActionToLegAngle_Tripod(self, motorcommands, actinons):
        """
        关节角度action∈[-1, 1]转换成实际关节角度[limit_min, limit_max]
        """
        # joint_angle = copy.deepcopy(motorcommands)
        # for i in range(9):
        #     if i%3 == 0:
        #         # joint_angle[i] = self.map_range(motorcommands[i], -1.0, 1.0, -actinons[i], actinons[i])
        #         joint_angle[i] = actinons[i//3] * motorcommands[i]
        #         joint_angle[i] = -joint_angle[i]
        #     elif i%3 == 1:
        #         joint_angle[i] = self.map_range(motorcommands[i], -0.0, 2.0, -actinons[i], actinons[i])
        #     else:
        #         joint_angle[i] = self.map_range(motorcommands[i], -0.0, 2.0, -actinons[i], actinons[i])
        #         # joint_angle[i] = 0.3
        # for i in range(9):
        #     if (i+9)%3 == 0:
        #         joint_angle[i+9] = self.map_range(motorcommands[i+9], -1.0, 1.0, -actinons[i+9], actinons[i+9])
                
        #     elif (i+9)%3 == 1:
        #         joint_angle[i+9] = self.map_range(motorcommands[i+9], -0.0, 2.0, -actinons[i+9], actinons[i+9])
        #     else:
        #         joint_angle[i+9] = self.map_range(motorcommands[i+9], -0.0, 2.0, -actinons[i+9], actinons[i+9])
        #         # joint_angle[i+9] = 0.3
        joint_angle = copy.deepcopy(motorcommands)
        for i in range(18):
            if i%3 == 0:
                # joint_angle[i] = self.map_range(motorcommands[i], -1.0, 1.0, actinons[i*4//3], actinons[i*4//3+1])
                # 直接学映射系数
                joint_angle[i] = actinons[i//3] * 0.6 * motorcommands[i]
                # 学映射系数变化值
                # joint_angle[i] = (actinons[i//3]+Initial_Action[i//3]) * motorcommands[i]
                # joint_angle[i] = (actinons[i//3]/3.0+Initial_Action[i//3]) * motorcommands[i]
            elif i%3 == 1:
                # joint_angle[i] = self.map_range(motorcommands[i], -0.0, 2.0, actinons[(i-1)*4//3+2], actinons[(i-1)*4//3+3])
                # 直接学映射系数
                joint_angle[i] = actinons[(i-1)//3 + 6] * 0.35 * motorcommands[i] + B2
                # 学映射系数变化值
                # joint_angle[i] = (actinons[(i-1)//3 + 6]+Initial_Action[(i-1)//3 + 6]) * motorcommands[i] + B2
                # joint_angle[i] = (actinons[(i-1)//3 + 6]/3.0+Initial_Action[(i-1)//3 + 6]) * motorcommands[i] + B2
            else:
                # joint_angle[i] = self.map_range(motorcommands[i], -0.0, 2.0, actinons[(i-2)*4//3+3], actinons[(i-2)*4//3+2])
                # 直接学映射系数
                joint_angle[i] = -actinons[(i-2)//3 + 6] * 0.35 * motorcommands[i] + B3
                # 学映射系数变化值
                # joint_angle[i] = -(actinons[(i-2)//3 + 6]+Initial_Action[(i-2)//3 + 6]) * motorcommands[i] + B3
                # joint_angle[i] = -(actinons[(i-2)//3 + 6]/3.0+Initial_Action[(i-2)//3 + 6]) * motorcommands[i] + B3
                # joint_angle[i] = 0.3
        return joint_angle

    def GetBaseHigh(self):
        return self.GetBasePosition()[2]

    def GetBaseOrientation(self):
        """Get the position of hexapod's base.
        """
        _, orientation = (self._pybullet_client.getBasePositionAndOrientation(self.my_phantomx))
        return orientation

    def GetBasePosition(self):
        """Get the position of hexapod's base.
        """
        position, _ = (self._pybullet_client.getBasePositionAndOrientation(self.my_phantomx))
        return position

    def GetBasePositionAndOrientation(self):
        """Get the position of body.
                                  [x,y,z,w]
            ((0.0, 0.0, 0.31), (0.0, 0.0, 0.0, 1.0))
        """
        position, orientation = (self._pybullet_client.getBasePositionAndOrientation(self.my_phantomx))
        return position, orientation
    
    def GetTrueMotorAngles(self):
        """Gets the motor angles at the current moment
        Returns:
            Motor angles
        """
        motor_angles = [self._pybullet_client.getJointState(self.my_phantomx, motor_id)[0] for motor_id in self._motor_id_list]
        motor_angles = np.array(motor_angles)
        self._joint_history_t2 = self._joint_history_t1 
        self._joint_history_t1 = self._joint_position
        self._joint_position = motor_angles

        return motor_angles
    
    def GetMotorAnglesHistoryT1(self):
        return self._joint_history_t1

    def GetMotorAnglesHistoryT2(self):
        return self._joint_history_t2

    def GetTrueMotorVelocities(self):
        """Get the velocity of all motors.

        Returns:
            Velocities of all motors.
        """
        motor_velocities = [
            self._pybullet_client.getJointState(self.my_phantomx, motor_id)[1]
            for motor_id in self._motor_id_list
        ]
        #motor_velocities = np.multiply(motor_velocities, self._motor_direction)
        return motor_velocities

    def GetTrueMotorTorques(self):
        """Get the amount of torque the motors are exerting.

        Returns:
        Motor torques of all motors.
        """
        motor_torques = [
            self._pybullet_client.getJointState(self.my_phantomx, motor_id)[3]
            for motor_id in self._motor_id_list
        ]
        return motor_torques
    
    def GetTrueBodyLinearVelocity(self):
        vel, _ = self._pybullet_client.getBaseVelocity(self.my_phantomx)
        return vel

    def GetTrueBodyAngularVelocity(self):
        _, yaw = self._pybullet_client.getBaseVelocity(self.my_phantomx)
        return yaw
    
    def GetBodyVelocity(self):
        vel_and_ang = self._pybullet_client.getBaseVelocity(self.my_phantomx)
        return vel_and_ang
    
    # 获取足底和地面的碰撞信息
    def GetCollisionWithGround(self):
        collision = self._pybullet_client.getContactPoints(self.my_phantomx)
        ankle_collision = [0] * 6
        # 只提取机器人ankle的碰撞index信息
        for i in range(6):
            for j in range(len(collision)):
                if collision[j][3] == self.joint_id[i*3+2]:
                    ankle_collision[i] = 1
                    
        # print("ankle_collision: ", ankle_collision)
        return ankle_collision

    def ApplyAction(self, motor_commands):
        current_joint_angles = self.GetTrueMotorAngles()
        motor_commands_max = (current_joint_angles + self.time_step * self._motor_velocity_limit)
        motor_commands_min = (current_joint_angles - self.time_step * self._motor_velocity_limit)
        motor_commands = np.clip(motor_commands, motor_commands_min, motor_commands_max)

        self._pybullet_client.setJointMotorControlArray(self.my_phantomx,
                                                        jointIndices = self.joint_id,
                                                        controlMode=self._pybullet_client.POSITION_CONTROL,
                                                        targetPositions = motor_commands,
                                                        forces = [30]*18,
                                                        positionGains=[1]*18,
                                                        velocityGains=[1]*18
                                                        )
                                                        