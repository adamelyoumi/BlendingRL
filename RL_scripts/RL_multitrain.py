# TB regex: (30/30_1021)|(33/33_1021)|(23/23_1021)|(32/32_1021)|(31/31_1021)|(29/29_1021)

import sys, os

curr_dir = os.path.abspath(os.getcwd())
sys.path.append(curr_dir)

import json
import numpy as np
import torch as th
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import safe_mean
from envs import BlendEnv, flatten_and_track_mappings, reconstruct_dict
from models import CustomMLP_ACP_simplest_softmax, CustomMLP_ACP_simplest_std
from utils import *

import yaml
import warnings
import datetime
import argparse



def params_equal(a, b):
    return all([th.all(t1 == t2).item() for t1, t2 in zip(a["policy"].values(), b["policy"].values())])

class CustomLoggingCallbackPPO(BaseCallback):
    def __init__(self, schedule_timesteps, start_log_std=2, end_log_std=-1, std_control = None):
        super().__init__(verbose = 0)
        self.std_control = std_control
        
        self.start_log_std = start_log_std
        self.end_log_std = end_log_std
        self.schedule_timesteps = schedule_timesteps
        self.current_step = 0
        self.perfs = []
        self.print_flag = False
        
        self.pen_M, self.pen_B, self.pen_P, self.pen_reg = [], [], [], []
        self.n_pen_M, self.n_pen_B, self.n_pen_P, self.pen_nv, self.pen_nv_counted = [], [], [], [], []
        self.units_sold, self.units_bought, self.rew_sold, self.rew_depth = [], [], [], []
        
    def _on_rollout_end(self) -> None:
        self.logger.record("penalties/in_out",              sum(self.pen_M)/len(self.pen_M))
        self.logger.record("penalties/buysell_bounds",      sum(self.pen_B)/len(self.pen_B))
        self.logger.record("penalties/tank_bounds",         sum(self.pen_P)/len(self.pen_P))
        
        self.logger.record("penalties/n_in_out",            sum(self.n_pen_M)/len(self.n_pen_M))
        self.logger.record("penalties/n_buysell_bounds",    sum(self.n_pen_B)/len(self.n_pen_B))
        self.logger.record("penalties/n_tank_bounds",       sum(self.n_pen_P)/len(self.n_pen_P))
        self.logger.record("penalties/n_vltn",              sum(self.pen_nv)/len(self.pen_nv))
        self.logger.record("penalties/n_vltn_counted",      sum(self.pen_nv_counted)/len(self.pen_nv_counted))
        
        self.logger.record("penalties/units_sold",          sum(self.units_sold)/len(self.units_sold))
        self.logger.record("penalties/units_bought",        sum(self.units_bought)/len(self.units_bought))
        self.logger.record("penalties/rew_sold",            sum(self.rew_sold)/len(self.rew_sold))
        self.logger.record("penalties/rew_depth",           sum(self.rew_depth)/len(self.rew_depth))
        
        self.perfs.append(safe_mean([ep_info["r"] for ep_info in model.ep_info_buffer]))
        
        self.pen_M, self.pen_B, self.pen_P, self.pen_reg = [], [], [], []
        self.n_pen_M, self.n_pen_B, self.n_pen_P, self.pen_nv, self.pen_nv_counted = [], [], [], [], []
        self.units_sold, self.units_bought, self.rew_sold, self.rew_depth = [], [], [], []
        
    def _on_step(self) -> bool:
        log_std: th.Tensor = self.model.policy.log_std
        t = self.locals["infos"][0]['dict_state']['t']
        
        if self.locals["infos"][0]["terminated"] or self.locals["infos"][0]["truncated"]: # record info at each episode end
            self.pen_M.append(self.locals["infos"][0]["pen_tracker"]["M"])
            self.pen_B.append(self.locals["infos"][0]["pen_tracker"]["B"])
            self.pen_P.append(self.locals["infos"][0]["pen_tracker"]["P"])
            
            self.n_pen_M.append(self.locals["infos"][0]["pen_tracker"]["n_M"])
            self.n_pen_B.append(self.locals["infos"][0]["pen_tracker"]["n_B"])
            self.n_pen_P.append(self.locals["infos"][0]["pen_tracker"]["n_P"])
            self.pen_nv.append(self.locals["infos"][0]["pen_tracker"]["n_M"] +
                               self.locals["infos"][0]["pen_tracker"]["n_B"] +
                               self.locals["infos"][0]["pen_tracker"]["n_P"])
            self.pen_nv_counted.append(self.locals["infos"][0]["pen_tracker"]["n_M"] if cfg["env"]["M"] > 0 else 0 +
                                       self.locals["infos"][0]["pen_tracker"]["n_B"] if cfg["env"]["B"] > 0 else 0 +
                                       self.locals["infos"][0]["pen_tracker"]["n_P"] if cfg["env"]["P"] > 0 else 0)
            # print(self.pen_nv, self.pen_nv_counted, self.n_pen_M, self.n_pen_B, self.n_pen_P)
            self.units_sold.append(self.locals["infos"][0]["pen_tracker"]["units_sold"])
            self.units_bought.append(self.locals["infos"][0]["pen_tracker"]["units_bought"])
            self.rew_sold.append(self.locals["infos"][0]["pen_tracker"]["rew_sold"])
            self.rew_depth.append(self.locals["infos"][0]["pen_tracker"]["rew_depth"])
        
        if self.num_timesteps%2048 < 6 and t == 1: # start printing
            self.print_flag = True
            
        # if self.print_flag:
        #     print("\nt:", t)
        #     if np.isnan(self.locals['rewards'][0]) or np.isinf(self.locals['rewards'][0]):
        #         print(f"is invalid reward {self.locals['rewards'][0]}")
        #     for i in ['obs_tensor', 'clipped_actions', 'rewards']:
        #         if i in self.locals:
        #             print(f"{i}: " + str(self.locals[i]))
        #     print(self.n_pen_M, self.n_pen_B, self.n_pen_P, self.n_pen_M, self.n_pen_B, self.n_pen_P,
        #           self.units_sold, self.units_bought, self.rew_sold, self.rew_depth)
        #     if t == 6:
        #         self.print_flag = False
        #         print(f"\n\nLog-Std at step {self.num_timesteps}: {log_std.detach().cpu().numpy()}")
        #         print(self.locals["infos"][0]["pen_tracker"])
        #         print("\n\n\n\n\n")
        
        if self.std_control:
            progress = self.current_step / self.schedule_timesteps
            new_log_std = self.start_log_std + progress * (self.end_log_std - self.start_log_std)
            self.model.policy.log_std.data.fill_(new_log_std)
            self.current_step += 1
        
        return True

# warnings.filterwarnings("ignore")




parser = argparse.ArgumentParser()
parser.add_argument("--configs")
parser.add_argument("--n_tries")
parser.add_argument("--n_timesteps")
parser.add_argument("--ignore_cfg_startingpoint", action='store_true')
parser.add_argument("--layout", default="simplest")

args = parser.parse_args()


CONFIGS = eval(args.configs) # list of int
N_TRIES = int(args.n_tries)
N_TIMESTEPS = int(args.n_timesteps)

connections, action_sample = get_jsons(args.layout)

sources, blenders, demands = get_sbp(connections)

if args.layout == "base":
    sigma = {"s1":{"q1": 0.06}, "s2":{"q1": 0.26}}
    sigma_ub = {"p1":{"q1": 0.16}, "p2":{"q1": 1}}
    sigma_lb = {"p1":{"q1": 0}, "p2":{"q1": 0}}
else:
    sigma = {s:{"q1": 0.06} for s in sources}       # No quality requirements
    sigma_ub = {d:{"q1": 0.16} for d in demands}    # No quality requirements
    sigma_lb = {d:{"q1": 0} for d in demands}       # No quality requirements
    
s_inv_lb = {s: 0 for s in sources}
s_inv_ub = {s: 999 for s in sources}
d_inv_lb = {d: 0 for d in demands}
d_inv_ub = {d: 999 for d in demands}
betaT_d = {d: 1 for d in demands} # Price of sold products
b_inv_ub = {j: 30 for j in blenders} 
b_inv_lb = {j: 0 for j in blenders}

old_params = None
best_model_sequential_train = {i: "" for i in CONFIGS}
# best_model_sequential_train = {
#     35: "35/model1",
#     36: "36/model3",
#     37: "37/model2"
# }

for id, train_id in enumerate(CONFIGS):
    perfs = {}
    # perfs = {
    #     "35/model1": [5,6,3],
    #     "35/model2": [4,6,2]
    # }
    for t in range(N_TRIES):
        try:
            with open(f"configs/{train_id}.yaml", "r") as f:
                s = "".join(f.readlines())
                cfg = yaml.load(s, Loader=yaml.FullLoader)
            
            betaT_s = {s: cfg["env"]["product_cost"]  for s in sources} # Cost of bought products
            
            T = 6
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

            env = BlendEnv(v = False, T = T,
                    D = cfg["env"]["D"], Q = cfg["env"]["Q"], 
                    P = cfg["env"]["P"], B = cfg["env"]["B"], 
                    Z = cfg["env"]["Z"], M = cfg["env"]["M"],
                    reg = cfg["env"]["reg"],
                    reg_lambda = cfg["env"]["reg_lambda"],
                    MAXFLOW = cfg["env"]["maxflow"],
                    alpha = cfg["env"]["alpha"], beta = cfg["env"]["beta"], 
                    max_pen_violations = cfg["env"]["max_pen_violations"],
                    illeg_act_handling = cfg["env"]["illeg_act_handling"],
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
                
            # optimizer_cls = eval(cfg["optimizer"])
            optimizer_cls = PPO

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

            kwa = dict(policy = policytype, env = env,
                        tensorboard_log = "./logs",
                        clip_range = cfg["model"]["clip_range"],
                        learning_rate = cfg["model"]["lr"],
                        ent_coef = cfg["model"]["ent_coef"],
                        use_sde = cfg["model"]["use_sde"],
                        batch_size = cfg["model"]["batch_size"],
                        policy_kwargs = policy_kwargs)

            model = optimizer_cls(**kwa)

            if args.ignore_cfg_startingpoint and id > 0:
                cfg_start = CONFIGS[id-1]
                try: # If starting point is an int, try to take the best model from current train
                    model.set_parameters(best_model_sequential_train[cfg_start])
                    print(f"Set parameters according to {best_model_sequential_train[cfg_start]}")
                except: # If no best model is available, take the most recent 
                    bin_ = get_bin(cfg_start)
                    directory = f"C:\\Users\\adame\\OneDrive\\Bureau\\CODE\\BlendingRL\\models\\{args.layout}\\{bin_}\\{cfg_start}"
                    chosen, mod_chosen = "", 0
                    for f in os.listdir(directory):
                        mod_time = os.path.getmtime(os.path.join(directory, f))
                        if mod_time > mod_chosen:
                            chosen = os.path.join("models", {args.layout}, {bin_}, {cfg_start}, f)
                    model.set_parameters(chosen)
                    print(f"Set parameters according to {chosen}")
                    
            elif args.ignore_cfg_startingpoint:
                print("Set parameters randomly")
                
            elif cfg["starting_point"]:
                try: # See if starting point is an int
                    cfg_start = int(cfg["starting_point"])
                    try: # If starting point is an int, try to take the best model from current train
                        model.set_parameters(best_model_sequential_train[cfg_start])
                        print(f"Set parameters according to {best_model_sequential_train[cfg_start]}")
                    except: # If no best model is available, take the most recent 
                        bin_ = get_bin(cfg_start)
                        directory = f"C:\\Users\\adame\\OneDrive\\Bureau\\CODE\\BlendingRL\\models\\{args.layout}\\{bin_}\\{cfg_start}"
                        chosen, mod_chosen = "", 0
                        for f in os.listdir(directory):
                            mod_time = os.path.getmtime(os.path.join(directory, f))
                            if mod_time > mod_chosen:
                                chosen = os.path.join("models", {args.layout}, {bin_}, {cfg_start}, f)
                        model.set_parameters(chosen)
                        print(f"Set parameters according to {chosen}")
                    
                except ValueError: # If not an int, it is a specific path
                    model.set_parameters(cfg["starting_point"])
                    print(f"Set parameters according to {cfg['starting_point']}")
            
            else:
                print("Set parameters randomly")
            
            
            
            if old_params is not None:
                t = params_equal(model.get_parameters(), old_params)
                if t: print("Model transfer OK !")
                else: print("######### Model transfer NOT OK ! #########", model.get_parameters(), "\n\n", old_params)
                
            bin_ = get_bin(cfg['id'])
            entcoef = str(model.ent_coef) if type(model) == PPO else ""
            cliprange = str(model.clip_range(0)) if type(model) == PPO else ""
            model_name = f"models/{args.layout}/{bin_}/{cfg['id']}/{cfg['id']}_{datetime.datetime.now().strftime('%m%d-%H%M%S')}"
            
            callback = CustomLoggingCallbackPPO(schedule_timesteps=N_TIMESTEPS, std_control = cfg["clipped_std"])

            print("Model name:", model_name)
            logpath = model_name[len("models/"):]
            print(f"Logging at logs/{logpath}")
            model.learn(total_timesteps = N_TIMESTEPS,
                        progress_bar = False,
                        tb_log_name = logpath,
                        callback = callback,
                        reset_num_timesteps = False)
            
            perfs[model_name] = sum(callback.perfs[-3:])/3 # Average reward over the last 10 rollouts

            model.save(model_name)
            
            del model
                    
        except Exception as e:
            raise e
            # continue
    
    maxperf = -1e6
    print("perfs:",perfs)
    for k, v in perfs.items():
        if v > maxperf:
            best_model_sequential_train[train_id] = k # Keeping the best performing model 
            
            
"""
- Continuity check & Imitation learning
- Sequential training
    - objective and in/out properly learned even on a 2/4/2 layout with 0/1 constraints, but the models just end up 
            un-learning the buy/sell rule or settle with "depth reward"
    - Why aren't some violation numbers starting at where the end of the previous config's curve is ?
    - We still observe models "lose progress" at some point: make the STD drop faster ? LR scheduling ?
    - Order of rules important ?
    - Discrepancy between amounts of product bought and sold
    - Why does config 37 learn the tank bounds rule and not the buy/sell bounds ?
- Try very big environment to see if the objective (buy/sell product) is still being learned ? No
"""