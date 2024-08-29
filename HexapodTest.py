import gym
# import torch as th
import torch
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from phantomx_env.envs.hexapod_env import PhantomxGymEnv

from src.CentralPatternGenerators.Hpof import PhantomxCPG
from src.AssistModulesCode.MatPlotAssitor import PlotModuleAssistor
from src.AssistModulesCode.ActionSelector import ActionModuleSelector
from src.CentralPatternGenerators.OnlineCPG import OnlinePhantomxCPG
from src.AssistModulesCode.Euler import quaternion_to_euler

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import random

TIME = 94*5*1
num_envs = 16
current_path = os.getcwd()

# 生成正太分布噪声
mu, sigma = 0, 0.01
s = np.random.normal(mu, sigma, 1000)
# print(s)
# random_value = s[random.randint(0, len(s) - 1)] 

def callable_env(env_id, kwargs):
    def aux():
        env = env_id(**kwargs)
        return env

    return aux

if __name__ == "__main__":
    multiprocessing.freeze_support()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # 100HZ控制帧率 + 非标准化环境
    tb_log_name = "PPO-TrackVelocity-100HZ-xvelP"

    if torch.cuda.is_available():
        # 获取当前使用的设备
        device = torch.cuda.current_device()
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("Using CPU for training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    torch.cuda.set_device(0)

    log_dir = current_path + "/"

    env = PhantomxGymEnv(render=True, set_goal_flag=True)
    # VecNormalize env read
    # env_kwargs = {"render": True, "set_goal_flag": True}
    # env = callable_env(PhantomxGymEnv, env_kwargs)
    # env = make_vec_env(env, n_envs=1)
    # env = VecNormalize.load(log_dir + "ppo_phantomx_trackvelEnv", env)
    # env.training = False
    # env.norm_reward = False

    model = PPO.load(log_dir + "ppo_hexapod_trackvel", env=env)
    # model = torch.load("model/ppo_model_2457600.pth")
    # model = PPO.load(log_dir + "zip/ppo_phantomx_trackvelLOG-37-2", env=env)
    # model = PPO.load(log_dir + "zip/ppo_phantomx_trackvelLOG-38", env=env)
    # model = PPO.load(log_dir + "model/modelVxLeakyRelu26_17600000", env=env)

    env.set_goal_state([0.21, 0.0, 0.0])
    obs = env.reset()

    # InitModules()
    PltModule = PlotModuleAssistor()
    CPGModule = PhantomxCPG()
    ActionModule = ActionModuleSelector()
    OnlineCPGModule = OnlinePhantomxCPG()

    TrueMotorAngle = []
    BaseOrientation = []
    TrueBodyVelocity = []
    TrueBodyAngVelocity = []
    TrueMotorVel = []
    Rewards = []
    Action = []
    Action2 = []

    # history_data = np.array([-1.0] * ((6 * 3) * 2)).reshape(1, -1)
    history_data = np.array([0.29616917,  1.04204406, -0.37516158, -1.01642539,  0.29616917,  1.04204406,
                        -0.37516158, -1.01642539,  0.29616917,  1.04204406, -0.37516158, -1.01642539,
                        -1.97166912,  0.36889423, -1.97166912,  0.36889423,  1.86400333, -0.74193836,
                        1.86400333, -0.74193836, -1.97166912,  0.36889423, -1.97166912,  0.36889423,
                        1.86400333, -0.74193836,  1.86400333, -0.74193836, -1.97166912,  0.36889423,
                        -1.97166912, 0.36889423,  1.86400333, -0.741193836,  1.86400333, -0.74193836]).reshape(1, -1)

# -----------------------------训练时注释该部分--------------------------------
    for i in range(TIME):
        action, _states = model.predict(obs)
        # # 机器人静止
        # action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # # 所有维度action为映射系数时的三足步态
        # action = [0.523599, 0.523599, 0.523599, -0.523599, -0.523599, -0.523599, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25]
        
        # # 给定action三足步态，髋关节为映射系数，膝、踝关节为映射系数变化量，检验模型直接注释这句代码。
        # action = [0.523599, 0.523599, 0.523599, -0.523599, -0.523599, -0.523599, 0, 0, 0, 0, 0, 0]
        # action = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1]
        # action = [0.5, 0.5, 0.5, -0.5, -0.5, -0.5, 0.25, 0.25, 0.25, -0.25, -0.25, -0.25]
        # action = [0.4, 0.4, 0.4, -0.4, -0.4, -0.4, 0, 0, 0, 0, 0, 0]

        # # 给action噪声
        # for j in range(12):
        #     action[j] = action[j] + s[random.randint(0, len(s) - 1)]
        # print("action", action)

        obs, rewards, dones, info = env.step(action)

        # 记录各种状态值便于后续绘图
        base_height = env.phantomx.GetBaseHigh()
        orientation = env.phantomx.GetBaseOrientation()
        base_position = env.phantomx.GetBasePosition()
        robot_linearvel = env.phantomx.GetTrueBodyLinearVelocity()
        robot_angularvel = env.phantomx.GetTrueBodyAngularVelocity()
        motor_vel = env.phantomx.GetTrueMotorVelocities()
        motor_torques = env.phantomx.GetTrueMotorTorques()

        _, _, yaw = quaternion_to_euler(orientation)
        
        motor_angle = env.phantomx.GetTrueMotorAngles()
        Rewards.append(rewards)
        TrueMotorAngle.append(motor_angle)
        Action.append(action[0])
        Action2.append(action[6])
        
        # if dones:
        #     env.reset()
        # env.render()

    # # 绘制各种observation的图像曲线
    # plt.figure()
    # plt.title("BaseOritentation")
    # plt.plot(BaseOrientation)
    # plt.legend()
    # plt.savefig(log_dir+"orientation")
    # plt.close()

    # plt.figure()
    # plt.title("TrueBodyVel")
    # plt.plot(TrueBodyVelocity)
    # plt.legend()
    # plt.savefig(log_dir+"bodyVel")
    # plt.close()

    # plt.figure()
    # plt.title("TrueAngVel")
    # plt.plot(TrueBodyAngVelocity)
    # plt.legend()
    # plt.savefig(log_dir+"bodyAng")
    # plt.close()

    # plt.figure()
    # plt.title("BaseOritentation")
    # plt.plot(BaseOrientation)
    # plt.legend()
    # plt.savefig(log_dir+"orientation")
    # plt.close()

    # plt.figure()
    # plt.title("MotorVel")
    # plt.plot(TrueMotorVel)
    # plt.legend()
    # plt.savefig(log_dir+"MotorVel")
    # plt.close()


    # 绘制关节角度随时间变化的图像
    TrueLegAngle = [ [] for _ in range(18) ]

    for i in range(TIME):
        for j in range(18):
            TrueLegAngle[j].append(TrueMotorAngle[i][j])

    # 绘制关节角、reward、action随时间变化的图像
    PltModule.plot(data=TrueLegAngle, plt_mode=0)
    PltModule.plot(data=TrueLegAngle, plt_mode=1)
    PltModule.plot(data=TrueLegAngle, plt_mode=2)
    PltModule.plot(data=Rewards, plt_mode=10)
    PltModule.plot(data=Action, plt_mode=12, save_name="HipAction")
    PltModule.plot(data=Action2, plt_mode=12, save_name="KneeAction")

    # # 在env中将除某一项奖励系数之外的其他项奖励系数设为0，可以用下面的代码绘制出该项reward的曲线
    # PltModule.plot(data=Rewards, plt_mode=11, save_name="YawReward")
    plt.close('all')

average_reward = np.mean(Rewards)
# TIME步数下的总的reward值和平均值
print("Rewards: ", sum(Rewards))
print("average_reward: ", average_reward)
# -----------------------------训练时注释该部分--------------------------------

    # step一个cpg改变副值