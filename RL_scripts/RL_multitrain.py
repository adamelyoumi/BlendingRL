
import sys, os

curr_dir = os.path.abspath(os.getcwd())
sys.path.append(curr_dir)

import json
import numpy as np
import torch as th
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecCheckNan
from envs import BlendEnv, flatten_and_track_mappings, reconstruct_dict
from models import CustomRNN_ACP, CustomMLP_ACP, CustomMLP_ACP_simplest_softmax, CustomMLP_ACP_simplest_std
from math import exp, log
import yaml
import warnings
import datetime
import argparse

def get_sbp(connections):
    sources = list(connections["source_blend"].keys())
    
    b_list = list(connections["blend_blend"].keys())
    for b in connections["blend_blend"].keys():
        b_list += connections["blend_blend"][b]
    b_list += list(connections["blend_demand"].keys())
    blenders = list(set(b_list))
    
    p_list = []
    for p in connections["blend_demand"].keys():
        p_list += connections["blend_demand"][p]
    demands = list(set(p_list))
    
    return sources, blenders, demands

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--configs")
parser.add_argument("--n_tries")
parser.add_argument("--n_timesteps")
parser.add_argument("--layout", default="simplest")


args = parser.parse_args()


CONFIGS = eval(args.configs)
N_TRIES = int(args.n_tries)
N_TIMESTEPS = int(args.n_timesteps)

with open(f"./configs/json/connections_{args.layout}.json" ,"r") as f:
    connections_s = f.readline()
connections = json.loads(connections_s)

with open(f"./configs/json/action_sample_{args.layout}.json" ,"r") as f:
    action_sample_s = f.readline()
action_sample = json.loads(action_sample_s)
action_sample_flat, mapp = flatten_and_track_mappings(action_sample)

sources, blenders, demands = get_sbp(connections)

if args.layout == "base":
    sigma = {"s1":{"q1": 0.06}, "s2":{"q1": 0.26}}
    sigma_ub = {"p1":{"q1": 0.16}, "p2":{"q1": 1}}
    sigma_lb = {"p1":{"q1": 0}, "p2":{"q1": 0}}
else:
    sigma = {s:{"q1": 0.06} for s in sources}
    sigma_ub = {d:{"q1": 0.16} for d in demands}
    sigma_lb = {d:{"q1": 0} for d in demands}
    
s_inv_lb = {s: 0 for s in sources}
s_inv_ub = {s: 999 for s in sources}
d_inv_lb = {d: 0 for d in demands}
d_inv_ub = {d: 999 for d in demands}
betaT_d = {d: 1 for d in demands} # Price of sold products
b_inv_ub = {j: 30 for j in blenders} 
b_inv_lb = {j: 0 for j in blenders}

for train_id in CONFIGS:
    
    for t in range(N_TRIES):
        try:
            with open(f"configs/{train_id}.yaml", "r") as f:
                s = "".join(f.readlines())
                cfg = yaml.load(s, Loader=yaml.FullLoader)
                
            class CustomLoggingCallbackPPO(BaseCallback):
                def __init__(self, schedule_timesteps, start_log_std=2, end_log_std=-1, std_control = cfg["clipped_std"]):
                    super().__init__(verbose = 0)
                    self.print_flag = False
                    self.std_control = std_control
                    
                    self.start_log_std = start_log_std
                    self.end_log_std = end_log_std
                    self.schedule_timesteps = schedule_timesteps
                    self.current_step = 0
                    
                    self.pen_M, self.pen_B, self.pen_P, self.pen_reg = [[]]*4
                    
                def _on_rollout_end(self) -> None:
                    self.logger.record("penalties/in_out", sum(self.pen_M)/len(self.pen_M))
                    self.logger.record("penalties/buysell_bounds", sum(self.pen_B)/len(self.pen_B))
                    self.logger.record("penalties/tank_bounds", sum(self.pen_P)/len(self.pen_P))
                    
                    self.pen_M, self.pen_B, self.pen_P, self.pen_reg = [], [], [], []
                    
                def _on_step(self) -> bool:
                    log_std: th.Tensor = self.model.policy.log_std
                    t = self.locals["infos"][0]['dict_state']['t']
                    
                    if self.locals["dones"][0]: # record info at each episode end
                        self.pen_M.append(self.locals["infos"][0]["pen_tracker"]["M"])
                        self.pen_B.append(self.locals["infos"][0]["pen_tracker"]["B"])
                        self.pen_P.append(self.locals["infos"][0]["pen_tracker"]["P"])
                    
                    if self.num_timesteps%2048 < 6 and t == 1: # start printing
                        self.print_flag = True
                        
                    if self.print_flag:
                        print("\nt:", t)
                        if np.isnan(self.locals['rewards'][0]) or np.isinf(self.locals['rewards'][0]):
                            print(f"is invalid reward {self.locals['rewards'][0]}")
                        for i in ['obs_tensor', 'actions', 'values', 'clipped_actions', 'new_obs', 'rewards']:
                            if i in self.locals:
                                print(f"{i}: " + str(self.locals[i]))
                        if t == 6:
                            self.print_flag = False
                            print(f"\n\nLog-Std at step {self.num_timesteps}: {log_std.detach().numpy()}")
                            print("\n\n\n\n\n")
                            
                    if self.std_control:
                        progress = self.current_step / self.schedule_timesteps
                        new_log_std = self.start_log_std + progress * (self.end_log_std - self.start_log_std)
                        self.model.policy.log_std.data.fill_(new_log_std)
                        self.current_step += 1
                            
                    return True

            betaT_s = {s: cfg["env"]["product_cost"]  for s in sources} # Cost of bought products
            if cfg["env"]["uniform_data"]:
                tau0   = {s: [np.random.binomial(1, 0.7) * np.random.normal(15, 2) for _ in range(13)] for s in sources}
                delta0 = {d: [np.random.binomial(1, 0.7) * np.random.normal(15, 2) for _ in range(13)] for d in demands}
            else:
                tau0   = {s: [10, 10, 10, 0, 0, 0] for s in sources}
                delta0 = {d: [0, 0, 0, 10, 10, 10] for d in demands}

            env = BlendEnv(v = False, 
                    D = cfg["env"]["D"], Q = cfg["env"]["Q"], 
                    P = cfg["env"]["P"], B = cfg["env"]["B"], 
                    Z = cfg["env"]["Z"], M = cfg["env"]["M"],
                    reg = cfg["env"]["reg"],
                    reg_lambda = cfg["env"]["reg_lambda"],
                    MAXFLOW = cfg["env"]["maxflow"],
                    alpha = cfg["env"]["alpha"],
                    beta = cfg["env"]["beta"],
                    connections = connections,
                    action_sample = action_sample,
                    tau0 = tau0,delta0 = delta0,
                    sigma = sigma,
                    sigma_ub = sigma_ub, sigma_lb = sigma_lb,
                    s_inv_lb = s_inv_lb, s_inv_ub = s_inv_ub,
                    d_inv_lb = d_inv_lb, d_inv_ub = d_inv_ub,
                    betaT_d = betaT_d, betaT_s = betaT_s,
                    b_inv_ub = b_inv_ub,
                    b_inv_lb = b_inv_lb)

            env = Monitor(env)
            env = DummyVecEnv([lambda: env])
            env = VecNormalize(env, norm_obs=cfg["obs_normalizer"], norm_reward=cfg["reward_normalizer"])
            
            if cfg["clipped_std"]:
                policytype = CustomMLP_ACP_simplest_std
            elif cfg["custom_softmax"]:
                policytype = CustomMLP_ACP_simplest_softmax
            elif cfg["policytype"] == "MLP":
                policytype = "MlpPolicy"
                
            optimizer_cls = eval(cfg["optimizer"])

            if cfg["model"]["act_fn"] == "ReLU":
                act_cls = th.nn.ReLU
            elif cfg["model"]["act_fn"] == "tanh":
                act_cls = th.nn.Tanh
            elif cfg["model"]["act_fn"] == "sigmoid":
                act_cls = th.nn.Sigmoid

            policy_kwargs = dict(
                net_arch=[dict(pi=[cfg["model"]["arch_layersize"]]*cfg["model"]["arch_n"], 
                               vf=[cfg["model"]["arch_layersize"]]*cfg["model"]["arch_n"])],
                activation_fn = act_cls,
                log_std_init = cfg["model"]["log_std_init"]
            )

            if optimizer_cls == PPO:
                kwa = dict(policy = policytype, 
                            env = env,
                            tensorboard_log = "./logs",
                            clip_range = cfg["model"]["clip_range"],
                            learning_rate = cfg["model"]["lr"],
                            ent_coef = cfg["model"]["ent_coef"],
                            use_sde = cfg["model"]["use_sde"],
                            batch_size = cfg["model"]["batch_size"],
                            policy_kwargs = policy_kwargs)
                
            else:
                kwa = dict(policy = policytype, 
                            env = env,
                            tensorboard_log = "./logs",
                            batch_size = cfg["model"]["batch_size"],
                            learning_rate = cfg["model"]["lr"])

            model = optimizer_cls(**kwa)

            if cfg["starting_point"]:
                model.set_parameters(cfg["starting_point"])
                
            bin_ = f"{(cfg['id']//12)*12 +1}-{(cfg['id']//12 +1)*12 }"
            entcoef = str(model.ent_coef) if type(model) == PPO else ""
            cliprange = str(model.clip_range(0)) if type(model) == PPO else ""
            model_name = f"models/simplest/{bin_}/{cfg['id']}/{cfg['id']}_{datetime.datetime.now().strftime('%m%d-%H%M')}"
            model_name
            
            callback = CustomLoggingCallbackPPO(schedule_timesteps=N_TIMESTEPS)

            print(f"logging at {model_name}")
            
            logpath = model_name[len("models/"):]
            print(f"logging at {logpath}")
            model.learn(total_timesteps = N_TIMESTEPS,
                        progress_bar = False,
                        tb_log_name = logpath,
                        callback = callback,
                        reset_num_timesteps = False)

            model.save(model_name)

            M,Q,P,B,Z,D = 0, 0, 0, 0, 1, 0
            env = BlendEnv(v = True, 
                        D = cfg["env"]["D"], Q = cfg["env"]["Q"], 
                        P = cfg["env"]["P"], B = cfg["env"]["B"], 
                        Z = cfg["env"]["Z"], M = cfg["env"]["M"],
                        reg = cfg["env"]["reg"],
                        reg_lambda = cfg["env"]["reg_lambda"],
                        MAXFLOW = cfg["env"]["maxflow"],
                        alpha = cfg["env"]["alpha"],
                        beta = cfg["env"]["beta"],
                        connections = connections,
                        action_sample = action_sample,
                        tau0 = tau0,delta0 = delta0,
                        sigma = sigma,
                        sigma_ub = sigma_ub, sigma_lb = sigma_lb,
                        s_inv_lb = s_inv_lb, s_inv_ub = s_inv_ub,
                        d_inv_lb = d_inv_lb, d_inv_ub = d_inv_ub,
                        betaT_d = betaT_d, betaT_s = betaT_s,
                        b_inv_ub = b_inv_ub,
                        b_inv_lb = b_inv_lb)
            env = Monitor(env)

            obs = env.reset()
            obs, obs_dict = obs
            for k in range(env.T):
                action, _ = model.predict(obs, deterministic=True)
                print("\n\n   ",reconstruct_dict(action, env.mapping_act))
                obs, reward, done, term, _ = env.step(action)
                dobs = reconstruct_dict(obs, env.mapping_obs)
                print("\n    >>     ",dobs["sources"], dobs["blenders"], dobs["demands"])
                print("   " ,reward)
                    
        except Exception as e:
            print(e)
            continue