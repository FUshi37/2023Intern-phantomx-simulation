import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
# import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import multiprocessing
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from phantomx_env.envs.phantomxCPG_Trakline_env_ppo37 import PhantomxGymEnv

from src.CentralPatternGenerators.Hpof import PhantomxCPG
from src.AssistModulesCode.MatPlotAssitor import PlotModuleAssistor
from src.AssistModulesCode.ActionSelector import ActionModuleSelector
from src.CentralPatternGenerators.OnlineCPG import OnlinePhantomxCPG
from src.AssistModulesCode.Euler import quaternion_to_euler

import argparse
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
from phantomx_env.envs.phantomxCPG_Trakline_env_ppo37 import PhantomxGymEnv
from ppo_continuous_action import Agent, make_env

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="HalfCheetahBulletEnv-v0",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default= 10000000,#2000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=12,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

def evaluate_policy(agent, env, num_episodes=10, render=False):
    rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if render:
                env.render()
        rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    avg_reward = np.mean(rewards)
    return avg_reward, rewards

def main(args):
    # Initialize the environment
    env = PhantomxGymEnv()
    envs = gym.vector.SyncVectorEnv(
        [make_env("HalfCheetahBulletEnv-v0", 1 + i, 0, False, "run_name") for i in range(1)]
    )

    # Load the trained model
    agent = Agent(envs=envs)
    agent.load(args.model_path)

    # Evaluate the policy
    avg_reward, rewards = evaluate_policy(agent, env, args.num_episodes, args.render)
    print(f"Average Reward over {args.num_episodes} episodes: {avg_reward}")

    # Plot the rewards
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Evaluation Rewards')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--num-episodes', type=int, default=10, help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    args = parser.parse_args()
    main(args)
