from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from envs import BlendEnv, flatten_and_track_mappings, reconstruct_dict
import numpy as np


def lr_scheduler(p):
    if p > 0.75:
        return 4e-3
    elif p > 0.5:
        return 2e-3
    else:
        return 1e-3

HP = {
    "ent_coef":[],
    "gamma": [],
    "clip_range":[0.5], 
    "learning_rate":[lr_scheduler],
    "Z":[],
    "M":[],
    "P":[]
}

for ent_coef in HP["ent_coef"]:
    for gamma in HP["gamma"]:
        for clip_range in HP["clip_range"]:
            for learning_rate in HP["learning_rate"]:
                for M in HP["M"]:
                    for P in HP["P"]:
                        for Z in HP["Z"]:   
                            env = BlendEnv(v = False, Z=Z, P=P, M=M)
                            env2 = Monitor(env)
                            model = PPO("MlpPolicy", env2, 
                                        ent_coef=ent_coef, gamma=gamma, clip_range=clip_range, learning_rate=learning_rate,
                                        verbose=1, tensorboard_log="./logs")
                            model.learn(total_timesteps=100000, progress_bar=False)
                            model.save(f"./models/model_ent_{ent_coef}_gam_{gamma}_clip_{clip_range}_lr_{learning_rate.__name__}_{M}_{Z}_{P}")
                            