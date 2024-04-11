import gym
# import torch as th
import torch
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from phantomx_env.envs.phantomxCPG_Trackline_env import PhantomxGymEnv

from src.CentralPatternGenerators.Hpof import PhantomxCPG
from src.AssistModulesCode.MatPlotAssitor import PlotModuleAssistor
from src.AssistModulesCode.ActionSelector import ActionModuleSelector
from src.CentralPatternGenerators.OnlineCPG import OnlinePhantomxCPG

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

TIME = 2000
num_envs = 16

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

class SaveModelCallback(BaseCallback):
    """
    Callback for saving a model (the policy) every `save_freq` steps
    """

    def __init__(self, save_freq: int, save_path: str, verbose=1):
        """
        :param save_freq: (int) How often to save the model
        :param save_path: (str) Path to the folder where the model will be saved
        :param verbose: (int) Verbosity level, 0: no print, 1: print training information
        """
        super(SaveModelCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(os.path.join(self.save_path + "model/", f"modelVxPPPOaction_{self.num_timesteps}.zip"))
        return True

def make_env():
    env = PhantomxGymEnv()
    return env
# def make_env

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

    log_dir = "/home/yangzhe/Intern/simulation/RL_phantomx_pybullet/"
# -----------------------------加载模型检验时注释该部分--------------------------------
    # env = PhantomxGymEnv()

    # env = make_vec_env(lambda: env, n_envs=num_envs, monitor_dir=log_dir + "tensorboard_log/", seed=0, wrapper_kwargs=dict(), vec_env_cls=SubprocVecEnv)
    
    # env = SubprocVecEnv(envs)

    env = SubprocVecEnv([make_env for _ in range(num_envs)])

    # env = VecNormalize(env, norm_obs=True)

    env = VecMonitor(env)

    callback = SaveModelCallback(save_freq=5e4, save_path=log_dir)

    policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                        net_arch=[dict(pi=[512, 256, 128], vf=[512 ,256, 128])])
    # policy_kwargs_SAC = dict(activation_fn=torch.nn.ReLU,
    #                          net_arch=[512, 256, 128])

    # model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs_SAC, verbose=1, tensorboard_log="./phantomx_tensorboard_test/")
    # model = PPO("MlpPolicy", env, device=device, policy_kwargs=policy_kwargs, learning_rate=2.5e-4, verbose=1, tensorboard_log="./phantomx_tensorboard_test/")
    # model = PPO.load(log_dir + "model/modelVxP2_22400000", env=env)
    model = PPO.load(log_dir + "ppo_phantomx_trackvel", env=env)
    # model = SAC.load(log_dir + "model/modelVxPSAC_1400000", env=env)
    model.learn(
        # total_timesteps=8192*20, reset_num_timesteps=True, tb_log_name="first_run"
        total_timesteps=8000*200, reset_num_timesteps=False, tb_log_name=tb_log_name, callback=callback
        # total_timesteps=8192*20, reset_num_timesteps=True, tb_log_name=tb_log_name
        # total_timesteps=8192*20, reset_num_timesteps=True, tb_log_name="first_run", callback=callback
    )

    model.save(log_dir + "ppo_phantomx_trackvel")
    # model.save(log_dir + "sac_phantomx_trackvel")

    print("Model saved!")

    del model, env
# -----------------------------加载模型检验时注释该部分--------------------------------
#     env = PhantomxGymEnv(render=True, set_goal_flag=True)

#     # model = PPO.load(log_dir + "ppo_phantomx_trackvel", env=env)
#     # model = PPO.load(log_dir + "model/modelVxPELU_19223552", env=env)
#     # model = PPO.load(log_dir + "model/modelVxPRelu10_31242752", env=env)
#     model = SAC.load(log_dir + "model/modelVxPSAC_1400000", env=env)

#     env.set_goal_state([0.52, 0.0, 0.0])

#     obs = env.reset()

#     # InitModules()
#     PltModule = PlotModuleAssistor()
#     CPGModule = PhantomxCPG()
#     ActionModule = ActionModuleSelector()
#     OnlineCPGModule = OnlinePhantomxCPG()

#     TrueMotorAngle = []
#     BaseOrientation = []
#     TrueBodyVelocity = []
#     TrueBodyAngVelocity = []
#     TrueMotorVel = []
#     Rewards = []
#     Action = []

#     # history_data = np.array([-1.0] * ((6 * 3) * 2)).reshape(1, -1)
#     history_data = np.array([0.29616917,  1.04204406, -0.37516158, -1.01642539,  0.29616917,  1.04204406,
#                         -0.37516158, -1.01642539,  0.29616917,  1.04204406, -0.37516158, -1.01642539,
#                         -1.97166912,  0.36889423, -1.97166912,  0.36889423,  1.86400333, -0.74193836,
#                         1.86400333, -0.74193836, -1.97166912,  0.36889423, -1.97166912,  0.36889423,
#                         1.86400333, -0.74193836,  1.86400333, -0.74193836, -1.97166912,  0.36889423,
#                         -1.97166912, 0.36889423,  1.86400333, -0.74193836,  1.86400333, -0.74193836]).reshape(1, -1)

# # -----------------------------训练时注释该部分--------------------------------
#     for i in range(TIME):
#         action, _states = model.predict(obs)
#         # action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#         # action = [0.523599, 0.523599, 0.523599, -0.523599, -0.523599, -0.523599, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25]
#         # action = [-0.523599, -0.523599, -0.523599, 0.523599, 0.523599, 0.523599, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25]
#         # action = [0.1, 0.1, 0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]
        
#         # print("action", action)
#         # env.setMotorCommand(motorcommands)

#         obs, rewards, dones, info = env.step(action=action)
#         # print("action:", action)
#         # print(rewards)
#         # print(dones)
#         base_height = env.phantomx.GetBaseHigh()
#         # print("base_height=", base_height)
#         orientation = env.phantomx.GetBaseOrientation()
#         base_position = env.phantomx.GetBasePosition()
#         robot_linearvel = env.phantomx.GetTrueBodyLinearVelocity()
#         robot_angularvel = env.phantomx.GetTrueBodyAngularVelocity()
#         motor_vel = env.phantomx.GetTrueMotorVelocities()
#         # print("motor_vel: ", motor_vel)
#         # print("linearvel: ", robot_linearvel)
#         # print("angvel: ", robot_angularvel)
#         # print("base_position: ", base_position)
#         # print("TrueMotroAngle: ", env.phantomx.GetTrueMotorAngles())
#         # print("orientation: ", orientation)
#         # sopActions = env.phantomx.ConvertActionToLegAngle_Tripod(action)
        
#         motor_angle = env.phantomx.GetTrueMotorAngles()
#         Rewards.append(rewards)
#         TrueMotorAngle.append(motor_angle)
#         Action.append(action[0])

#         # BaseOrientation.append(orientation)
#         # TrueBodyVelocity.append(robot_linearvel)
#         # TrueBodyAngVelocity.append(robot_angularvel)
#         # TrueMotorVel.append(motor_vel)
        
#         # if dones:
#         #     env.reset()
#         # env.render()

#     # # 绘制各种observation的图像曲线
#     # plt.figure()
#     # plt.title("BaseOritentation")
#     # plt.plot(BaseOrientation)
#     # plt.legend()
#     # plt.savefig(log_dir+"orientation")
#     # plt.close()

#     # plt.figure()
#     # plt.title("TrueBodyVel")
#     # plt.plot(TrueBodyVelocity)
#     # plt.legend()
#     # plt.savefig(log_dir+"bodyVel")
#     # plt.close()

#     # plt.figure()
#     # plt.title("TrueAngVel")
#     # plt.plot(TrueBodyAngVelocity)
#     # plt.legend()
#     # plt.savefig(log_dir+"bodyAng")
#     # plt.close()

#     # plt.figure()
#     # plt.title("BaseOritentation")
#     # plt.plot(BaseOrientation)
#     # plt.legend()
#     # plt.savefig(log_dir+"orientation")
#     # plt.close()

#     # plt.figure()
#     # plt.title("MotorVel")
#     # plt.plot(TrueMotorVel)
#     # plt.legend()
#     # plt.savefig(log_dir+"MotorVel")
#     # plt.close()


#     # 绘制关节角度随时间变化的图像
#     TrueLegAngle = [ [] for _ in range(18) ]

#     for i in range(TIME):
#     # for i in range(100):
#         for j in range(18):
#             TrueLegAngle[j].append(TrueMotorAngle[i][j])

#     PltModule.plot(data=TrueLegAngle, plt_mode=0)
#     PltModule.plot(data=TrueLegAngle, plt_mode=1)
#     PltModule.plot(data=TrueLegAngle, plt_mode=2)
#     PltModule.plot(data=Rewards, plt_mode=10)
#     PltModule.plot(data=Action, plt_mode=12)
#     # PltModule.plot(data=Rewards, plt_mode=11, save_name="ForwardReward")
#     plt.close('all')
# -----------------------------训练时注释该部分--------------------------------

    #todo
    # 改action space
    # 改reward增加角速度
    # 用pybullet自带的速度函数
    # 改observation去掉position

    # step一个cpg改变副值