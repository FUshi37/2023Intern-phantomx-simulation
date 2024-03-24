import gym
import torch as th
from stable_baselines3 import PPO, DQN, DDPG
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from phantomx_env.envs.phantomx_env import PhantomxGymEnv
from src.CentralPatternGenerators.Hpof import PhantomxCPG
from src.AssistModulesCode.MatPlotAssitor import PlotModuleAssistor
from src.AssistModulesCode.ActionSelector import ActionModuleSelector
from src.CentralPatternGenerators.OnlineCPG import OnlinePhantomxCPG

import os
import numpy as np
import matplotlib.pyplot as plt

TIME = 3000


def SaveData(data, filename, directory="data"):
    for i in range(len(data)):
        file_path = os.path.join(directory, filename + "-" + str(i) + ".npy")
        np.save(file_path, data[i])

def LoadData(filename, directory="data"):
    data = []
    for i in range(18):
        file_path = os.path.join(directory, filename + "-" + str(i) + ".npy")
        data.append(np.load(file_path))
    return data

# env = PhantomxGymEnv()

# policy_kwargs = dict(activation_fn=th.nn.ReLU,
#                      net_arch=[dict(pi=[256, 128], vf=[256, 128])])
# model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, learning_rate=2.5e-4, verbose=1, tensorboard_log="./phantomx_tensorboard_test/")

# model.learn(
#     total_timesteps=6144*180, tb_log_name="first_run"
# )
log_dir = "/home/yangzhe/Intern/simulation/RL_phantomx_pybullet/"
# model.save(log_dir + "ppo_phantomx")

# print("Model saved!")

# del model, env

env = PhantomxGymEnv(render=True)

# model = PPO.load(log_dir + "ppo_phantomx", env=env)
# model = PPO.load(log_dir + "zip/ppo_phantomx_jump", env=env)

obs = env.reset()

# InitModules()
PltModule = PlotModuleAssistor()
CPGModule = PhantomxCPG()
ActionModule = ActionModuleSelector()
OnlineCPGModule = OnlinePhantomxCPG()

# t = np.arange(0, 100, 0.01)
# data = CPGModule.calculate(t)
# leg1_hip, leg1_knee, leg1_ankle = CPGModule.TripodGait(1, t, data)
# leg2_hip, leg2_knee, leg2_ankle = CPGModule.TripodGait(2, t, data)
# leg3_hip, leg3_knee, leg3_ankle = CPGModule.TripodGait(3, t, data)
# leg4_hip, leg4_knee, leg4_ankle = CPGModule.TripodGait(4, t, data)
# leg5_hip, leg5_knee, leg5_ankle = CPGModule.TripodGait(5, t, data)
# leg6_hip, leg6_knee, leg6_ankle = CPGModule.TripodGait(6, t, data)

# Data = [leg1_hip, leg1_knee, leg1_ankle, leg2_hip, leg2_knee, leg2_ankle, leg3_hip, leg3_knee, leg3_ankle,
#         leg4_hip, leg4_knee, leg4_ankle, leg5_hip, leg5_knee, leg5_ankle, leg6_hip, leg6_knee, leg6_ankle]
# SaveData(Data, "90deg/90deg")
LegData = LoadData("90deg/90deg")
TrueMotorAngle = []

# history_data = np.array([-1.0] * ((6 * 3) * 2)).reshape(1, -1)
history_data = np.array([0.29616917,  1.04204406, -0.37516158, -1.01642539,  0.29616917,  1.04204406,
                    -0.37516158, -1.01642539,  0.29616917,  1.04204406, -0.37516158, -1.01642539,
                    -1.97166912,  0.36889423, -1.97166912,  0.36889423,  1.86400333, -0.74193836,
                    1.86400333, -0.74193836, -1.97166912,  0.36889423, -1.97166912,  0.36889423,
                    1.86400333, -0.74193836,  1.86400333, -0.74193836, -1.97166912,  0.36889423,
                    -1.97166912, 0.36889423,  1.86400333, -0.74193836,  1.86400333, -0.74193836]).reshape(1, -1)


for i in range(TIME):
    # action, _states = model.predict(obs)
    
    # online CPG
    t = np.linspace(i*0.01, i*0.01+0.01, 2)
    data = OnlineCPGModule.online_calculate(t, initial_values=history_data[-1, :])
    if (len(data)==2):
        data = data[1:].reshape(1, -1)
    history_data = np.vstack((history_data, data))

    action = ActionModule.SelectAction(action_mode=10, t=t, data=data[-1, :])
    # onlie CPG end

    # action = ActionModule.SelectAction(action_mode=0, aim_leg_pos=LegData, index=i, command_start_index=0)
    # action = ActionModule.SelectAction(action_mode=-1, aim_leg_pos=LegData, index=i, command_start_index=5000, TOTAL_TIME=TIME)
    # action = ActionModule.SelectAction(action_mode=1)
    
    print("action", action)
    obs, rewards, dones, info = env.step(action)
    print(rewards)
    print(dones)
    base_height = env.phantomx.GetBaseHigh()
    print("base_height=", base_height)
    orientation = env.phantomx.GetBaseOrientation()
    print("TrueMotroAngle: ", env.phantomx.GetTrueMotorAngles())
    print("orientation: ", orientation)
    sopActions = env.phantomx.ConvertActionToLegAngle_Tripod(action)
    
    motor_angle = env.phantomx.GetTrueMotorAngles()
    TrueMotorAngle.append(motor_angle)

    env.render()

# 绘制关节角度随时间变化的图像
TrueLegAngle = [ [] for _ in range(18) ]

for i in range(TIME):
    for j in range(18):
        TrueLegAngle[j].append(TrueMotorAngle[i][j])

PltModule.plot(data=TrueLegAngle, plt_mode=0)


