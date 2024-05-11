from torch import nn
from torch.functional import *
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from envs import BlendEnv

env = BlendEnv(alpha = 0.3, beta = 0)

# Wrap the environment in a vectorized environment
vec_env = DummyVecEnv([lambda: env])

# Define and train the PPO agent
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=100) 

model.save("ppo_myenv_model")

# Optionally, evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")