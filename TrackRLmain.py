import gym
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
from src.AssistModulesCode.Euler import quaternion_to_euler

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

TIME = 94*5*1
# 4G显存
num_envs = 16
current_path = os.getcwd()

def callable_env(env_id, kwargs):
    def aux():
        env = env_id(**kwargs)
        return env

    return aux

# 定步长保存模型
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
            self.model.save(os.path.join(self.save_path + "model/", f"modelVxLeakyRelu31_{self.num_timesteps}.zip"))
        return True

def make_env():
    env = PhantomxGymEnv()
    return env

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
# -----------------------------加载模型检验时注释该部分--------------------------------

    env = SubprocVecEnv([make_env for _ in range(num_envs)])

    env = VecMonitor(env)
    
    # 定步长保存模型
    callback = SaveModelCallback(save_freq=5e4, save_path=log_dir)

    policy_kwargs = dict(activation_fn=torch.nn.LeakyReLU,
                        net_arch=[dict(pi=[512, 256, 128], vf=[512 ,256, 128])])
    # policy_kwargs = dict(activation_fn=torch.nn.Tanh,
    #                     net_arch=[dict(pi=[512, 256, 128], vf=[512 ,256, 128])])
    # policy_kwargs = dict(activation_fn=torch.nn.Tanh,
    #                     net_arch=[dict(pi=[128, 256, 512, 256, 128], vf=[128, 256, 512, 256, 128])])
    # policy_kwargs_SAC = dict(activation_fn=torch.nn.ReLU,
    #                          net_arch=[512, 256, 128])

    # ppo_config = {  "gamma":0.99, 
    #                 "n_steps": 2048, 
    #                 "ent_coef":0.00,  
    #                 "learning_rate":1e-4, 
    #                 "vf_coef":0.5,
    #                 "max_grad_norm":0.5, 
    #                 "gae_lambda":0.95, 
    #                 "batch_size":128,
    #                 "n_epochs":10, 
    #                 "clip_range":0.2, 
    #                 "clip_range_vf":1,
    #                 "verbose":1, 
    #                 "tensorboard_log":"./phantomx_tensorboard_test/", 
    #                 "_init_setup_model":True, 
    #                 "policy_kwargs":policy_kwargs,
    #                 "device": device}
    
    ppo_config = {  "learning_rate": 2.5e-4,
                    "verbose":1, 
                    "tensorboard_log":"./phantomx_tensorboard_test/",
                    "policy_kwargs":policy_kwargs,
                    "device": device }

    model = PPO("MlpPolicy", env, **ppo_config)
    # model = PPO.load(log_dir + "ppo_phantomx_trackvel", env=env)
    model.learn(
        total_timesteps=16*94*5*1600, reset_num_timesteps=True, tb_log_name=tb_log_name, callback=callback
    )

    model.save(log_dir + "ppo_phantomx_trackvel")
    # env.save(log_dir + "ppo_phantomx_trackvelEnv")
    # model.save(log_dir + "sac_phantomx_trackvel")

    print("Model saved!")

    del model, env