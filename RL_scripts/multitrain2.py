
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))
try:
    print(curr_dir)
except:
    curr_dir = os.path.dirname(os.path.abspath(os.getcwd()))
    os.chdir(curr_dir)
    print(curr_dir)

# %%
import json
import numpy as np
import torch as th
from stable_baselines3 import PPO, DDPG, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import *
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecCheckNan

from envs import BlendEnv, flatten_and_track_mappings, reconstruct_dict
from models import *
from math import exp, log
import yaml

import warnings
warnings.filterwarnings("ignore")

# %%
with open("./configs/32.yaml", "r") as f:
    s = "".join(f.readlines())
    cfg = yaml.load(s, Loader=yaml.FullLoader)
    
layout = "simplest"

# %%
if cfg["custom_softmax"]:
    policytype = CustomMLP_ACP_simplest_softmax
elif cfg["policytype"] == "MLP":
    policytype = "MlpPolicy"
elif cfg["policytype"] == "MLPtanh":
    policytype = CustomMLP_ACP_simplest_tanh
    
optimizer_cls = eval(cfg["optimizer"])

if cfg["model"]["act_fn"] == "ReLU":
    act_cls = th.nn.ReLU
elif cfg["model"]["act_fn"] == "tanh":
    act_cls = th.nn.Tanh
elif cfg["model"]["act_fn"] == "sigmoid":
    act_cls = th.nn.Sigmoid

# %%
with open(f"./configs/json/connections_{layout}.json" ,"r") as f:
    connections_s = f.readline()
connections = json.loads(connections_s)

with open(f"./configs/json/action_sample_{layout}.json" ,"r") as f:
    action_sample_s = f.readline()
action_sample = json.loads(action_sample_s)
action_sample_flat, mapp = flatten_and_track_mappings(action_sample)

# %%
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
sources, blenders, demands = get_sbp(connections)

# %%
T = 6
if layout == "base":
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
betaT_s = {s: cfg["env"]["product_cost"]  for s in sources} # Cost of bought products

if cfg["env"]["uniform_data"]:
    if cfg["env"]["max_pen_violations"] < 999:
        max_ep_length = 50
        tau0   = {s: [np.random.normal(20, 3) for _ in range(max_ep_length)] for s in sources}
        delta0 = {d: [np.random.normal(20, 3) for _ in range(max_ep_length)] for d in demands}
        T = max_ep_length
        
    else:
        tau0   = {s: [np.random.normal(20, 3) for _ in range(13)] for s in sources}
        delta0 = {d: [np.random.normal(20, 3) for _ in range(13)] for d in demands}
else:
    tau0   = {s: [10, 10, 10, 0, 0, 0] for s in sources}
    delta0 = {d: [0, 0, 0, 10, 10, 10] for d in demands}

# %%
env = BlendEnv(v = False, T = T,
               D = cfg["env"]["D"], Q = cfg["env"]["Q"], P = cfg["env"]["P"], B = cfg["env"]["B"], Z = cfg["env"]["Z"], M = cfg["env"]["M"],
               reg = cfg["env"]["reg"], reg_lambda = cfg["env"]["reg_lambda"],
               MAXFLOW = cfg["env"]["maxflow"], alpha = cfg["env"]["alpha"], 
               beta = cfg["env"]["beta"], max_pen_violations = cfg["env"]["max_pen_violations"], connections = connections, 
               action_sample = action_sample, tau0 = tau0, delta0 = delta0, sigma = sigma,
               sigma_ub = sigma_ub, sigma_lb = sigma_lb,
               s_inv_lb = s_inv_lb, s_inv_ub = s_inv_ub,
               d_inv_lb = d_inv_lb, d_inv_ub = d_inv_ub,
               betaT_d = betaT_d, betaT_s = betaT_s,
               b_inv_ub = b_inv_ub,
               b_inv_lb = b_inv_lb)

# %%
env = Monitor(env)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, 
                   norm_obs=cfg["obs_normalizer"], 
                   norm_reward=cfg["reward_normalizer"])
# env = VecCheckNan(env, raise_exception=True)

# %%
policy_kwargs = dict(
    net_arch=[dict(pi = [cfg["model"]["arch_layersize"]] * cfg["model"]["arch_n"], 
                   vf = [cfg["model"]["arch_layersize"]] * cfg["model"]["arch_n"])],
    activation_fn = act_cls,
    log_std_init = cfg["model"]["log_std_init"]
)

# %%
print(policytype)

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

# %% [markdown]
# If batch_size = 64 and n_steps = 2048, then 1 epoch = 2048/64 = 32 batches

# %%
import datetime

bin_ = f"{(cfg['id']//12)*12 +1}-{(cfg['id']//12 +1)*12 }"
entcoef = str(model.ent_coef) if type(model) == PPO else ""
cliprange = str(model.clip_range(0)) if type(model) == PPO else ""
model_name = f"models/{layout}/{bin_}/{cfg['id']}/{cfg['id']}_{datetime.datetime.now().strftime('%m%d-%H%M')}"
model_name

# %%
class CustomLoggingCallbackPPO(BaseCallback):
    def __init__(self, schedule_timesteps, start_log_std=2, end_log_std=-1, std_control = cfg["clipped_std"]):
        super().__init__(verbose = 0)
        self.print_flag = False
        self.std_control = std_control
        
        self.start_log_std = start_log_std
        self.end_log_std = end_log_std
        self.schedule_timesteps = schedule_timesteps
        self.current_step = 0
        
        self.pen_M, self.pen_B, self.pen_P, self.pen_reg, self.pen_nv = [], [], [], [], []
        self.n_pen_M, self.n_pen_B, self.n_pen_P = [], [], []
        
    def _on_rollout_end(self) -> None:
        self.logger.record("penalties/in_out",              sum(self.pen_M)/len(self.pen_M))
        self.logger.record("penalties/buysell_bounds",      sum(self.pen_B)/len(self.pen_B))
        self.logger.record("penalties/tank_bounds",         sum(self.pen_P)/len(self.pen_P))
        self.logger.record("penalties/n_in_out",            sum(self.n_pen_M)/len(self.n_pen_M))
        self.logger.record("penalties/n_buysell_bounds",    sum(self.n_pen_B)/len(self.n_pen_B))
        self.logger.record("penalties/n_tank_bounds",       sum(self.n_pen_P)/len(self.n_pen_P))
        self.logger.record("penalties/n_vltn",              sum(self.pen_nv)/len(self.pen_nv))
        
        self.pen_M, self.pen_B, self.pen_P, self.pen_reg, self.pen_nv = [], [], [], [], []
        self.n_pen_M, self.n_pen_B, self.n_pen_P = [], [], []
        
    def _on_step(self) -> bool:
        
        log_std: th.Tensor = self.model.policy.log_std
        t = self.locals["infos"][0]['dict_state']['t']
        
        if self.locals["infos"][0]["terminated"] or self.locals["infos"][0]["truncated"]: # record info at each episode end
            self.pen_M.append(self.locals["infos"][0]["pen_tracker"]["M"])
            self.pen_B.append(self.locals["infos"][0]["pen_tracker"]["B"])
            self.pen_P.append(self.locals["infos"][0]["pen_tracker"]["P"])
            self.n_pen_M.append(-self.locals["infos"][0]["pen_tracker"]["M"]/cfg["env"]["M"])
            self.n_pen_B.append(-self.locals["infos"][0]["pen_tracker"]["B"]/cfg["env"]["P"])
            self.n_pen_P.append(-self.locals["infos"][0]["pen_tracker"]["P"]/cfg["env"]["B"])
            self.pen_nv.append(self.locals["infos"][0]["pen_tracker"]["n_violations"])
        
        if self.num_timesteps%2048 < 10 and t == 1: # start printing
            self.print_flag = True
            
        if self.print_flag:
            print("\nt:", t)
            if np.isnan(self.locals['rewards'][0]) or np.isinf(self.locals['rewards'][0]):
                print(f"is invalid reward {self.locals['rewards'][0]}")
            for i in ['obs_tensor', 'actions', 'values', 'clipped_actions', 'new_obs', 'rewards']:
                if i in self.locals:
                    print(f"{i}: " + str(self.locals[i]))
            if t == 10:
                self.print_flag = False
                print(f"\n\nLog-Std at step {self.num_timesteps}: {log_std.detach().cpu().numpy()}")
                print("\n\n\n\n\n")
                
        if self.std_control:
            progress = self.current_step / self.schedule_timesteps
            new_log_std = self.start_log_std + progress * (self.end_log_std - self.start_log_std)
            self.model.policy.log_std.data.fill_(new_log_std)
            self.current_step += 1
                
        return True

# %%
total_timesteps = 1e5
log_callback = CustomLoggingCallbackPPO(schedule_timesteps=total_timesteps) if optimizer_cls == PPO else CustomLoggingCallbackDDPG()
callback = CallbackList([log_callback])
model_name

# %%
logpath = model_name[len("models/"):]
print(f"logging at {logpath}")
model.learn(total_timesteps = total_timesteps,
            progress_bar = False,
            tb_log_name = logpath,
            callback = callback,
            reset_num_timesteps = False)

# %%
import re

def save_next_file(directory, model_name):
    base_pattern = re.compile(model_name + r"_(\d+)\.zip")
    
    try:
        files = os.listdir(directory)
    except:
        os.mkdir(directory)
        
        files = os.listdir(directory)
        
    max_number = 0
    for file in files:
        match = base_pattern.match(file)
        if match:
            number = int(match.group(1))
            max_number = max(max_number, number)
    
    # Generate the next filename
    next_file_number = max_number + 1
    next_file_name = f"{model_name}_{next_file_number}"
    next_file_path = os.path.join(directory, next_file_name)
    
    model.save(next_file_path)
    
save_next_file(os.path.dirname(model_name), os.path.basename(model_name) )
