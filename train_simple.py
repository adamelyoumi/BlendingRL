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

warnings.filterwarnings("ignore")

class CustomLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.log_stds = []
        self.total_rewards = []
        self.signal = True
        self.update1 = True
        self.print_flag = False

    def _on_step(self) -> bool:
        log_std: th.Tensor = self.model.policy.log_std
        t = self.locals["infos"][0]['dict_state']['t']
        
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
                stds = th.exp(self.model.policy.log_std).mean().item()
        
                # if stds > 50:
                #     print("clipping log-stds")
                #     print("before: ", self.model.policy.log_std)
                #     self.model.policy.log_std = nn.Parameter( 2*th.ones(self.model.policy.log_std.shape, requires_grad=True) )
                #     print("after: ",  self.model.policy.log_std)
                    
                self.logger.record('train/learning_rate', self.model.learning_rate)
                self.logger.record('train/clip_range', self.model.clip_range(0))
                self.logger.record("train/std", th.exp(self.model.policy.log_std).mean().item())
                
                # if self.locals['rewards'][0] > 200 and self.update1:
                #     # self.training_env.set_attr('', 1e2)
                #     self.model.learning_rate = 1e2
                #     self.model.clip_range = 5e-2
                #     self.update1 = False
                
                self.print_flag = False
                print(f"\n\nLog-Std at step {self.num_timesteps}: {log_std.detach().numpy()}")
                self.log_stds.append(log_std.mean().item())
                self.total_rewards.append(self.locals['rewards'][0])
                print(f"\nAvg rewards over the last 100 episodes:{sum(self.total_rewards[-100:])/100} ; last reward: {self.total_rewards[-1]}")
                self.model.learning_rate
                print("\n\n\n\n\n\n")
                
        return True

CONFIGS = [3,12]
N_TRIES = 5
N_TIMESTEPS = 150000

connections = {
    "source_blend": {
        "s1": ["j1", "j2", "j3", "j4"],
        "s2": ["j1", "j2", "j3", "j4"]
    },
    "blend_blend": {"j1": [], "j2": [], "j3": [], "j4": []},
    "blend_demand": {"j1": ["p1", "p2"],
        "j2": ["p1", "p2"],
        "j3": ["p1", "p2"],
        "j4": ["p1", "p2"]
    }
}

action_sample = {
    'source_blend':{
        's1': {'j1':1, 'j2':1, 'j3':1, 'j4':0}, # From s1 to b1, from s1 to b2 etc...
        's2': {'j1':1, 'j2':1, 'j3':0, 'j4':1},
    },
    'blend_blend':{},
    'blend_demand':{
        'j1': {'p1':1, 'p2':0},
        'j2': {'p1':1, 'p2':2},
        'j3': {'p1':1, 'p2':2},
        'j4': {'p1':1, 'p2':2}
    },
    "tau": {"s1": 10, "s2": 10},
    "delta": {"p1": 0, "p2": 0}
}
action_sample_flat, _ = flatten_and_track_mappings(action_sample)

action_sample_flat, mapp = flatten_and_track_mappings(action_sample)

tau0   = {'s1': [10, 10, 10, 0, 0, 0], 's2': [10, 10, 10, 0, 0, 0]}
delta0 = {'p1': [0, 0, 0, 10, 10, 10], 'p2': [0, 0, 0, 10, 10, 10]}
sigma = {"s1":{"q1": 0.06}, "s2":{"q1": 0.26}} # Source concentrations
sigma_ub = {"p1":{"q1": 0.16}, "p2":{"q1": 0.16}} # Demand concentrations UBs/LBs
sigma_lb = {"p1":{"q1": 0}, "p2":{"q1": 0}}
s_inv_lb = {'s1': 0, 's2': 0}
s_inv_ub = {'s1': 999, 's2': 999}
d_inv_lb = {'p1': 0, 'p2': 0}
d_inv_ub = {'p1': 999, 'p2': 999}
betaT_d = {'p1': 1, 'p2': 1} # Price of sold products
b_inv_ub = {"j1": 30, "j2": 30, "j3": 30, "j4": 30} 
b_inv_lb = {j:0 for j in b_inv_ub.keys()}


for train_id in CONFIGS:
    for t in range(N_TRIES):
        try:
            with open(f"configs/{train_id}.yaml", "r") as f:
                s = "".join(f.readlines())
                cfg = yaml.load(s, Loader=yaml.FullLoader)
                
            betaT_s = {'s1': cfg["env"]["product_cost"], 's2': cfg["env"]["product_cost"]} # Cost of bought products

            env = BlendEnv(v = False, 
                        D = cfg["env"]["D"], 
                        Q = cfg["env"]["Q"], 
                        P = cfg["env"]["P"], 
                        B = cfg["env"]["B"], 
                        Z = cfg["env"]["Z"], 
                        M = cfg["env"]["M"],
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
            env = VecNormalize(env, 
                            norm_obs=cfg["obs_normalizer"], 
                            norm_reward=cfg["reward_normalizer"])
            env = VecCheckNan(env, raise_exception=True)

            policy_kwargs = dict(
                net_arch=[dict(pi=[cfg["model"]["arch_layersize"]]*cfg["model"]["arch_n"], 
                            vf=[cfg["model"]["arch_layersize"]]*cfg["model"]["arch_n"])],
                activation_fn = th.nn.ReLU,
                log_std_init = cfg["model"]["log_std_init"]
            )

            if cfg["clipped_std"]:
                policytype = CustomMLP_ACP_simplest_std
            elif cfg["custom_softmax"]:
                policytype = CustomMLP_ACP_simplest_softmax
            elif cfg["policytype"] == "MLP":
                policytype = "MlpPolicy"
                
            if cfg["optimizer"] == "PPO":
                optimizer_cls = PPO
            elif cfg["optimizer"] == "DDPG":
                optimizer_cls = DDPG
                
            model = optimizer_cls(policytype, 
                                env,
                                tensorboard_log = "./logs",
                                clip_range = cfg["model"]["clip_range"],
                                learning_rate = cfg["model"]["lr"],
                                ent_coef = cfg["model"]["ent_coef"],
                                policy_kwargs = policy_kwargs)

            # model.set_parameters("models\\simplest_model_0606-1629_ent_0.001_gam_0.99_clip_0.3_1000_1000_0")

            modeltype = "PPO" if type(model) == PPO else "DDPG"
            if type(model.policy) == CustomRNN_ACP:
                policytype = "CRNN"
            elif type(model.policy) == CustomMLP_ACP_simplest_std:
                policytype = "CMLP"
            else:
                policytype = "MLP"
                
            entcoef = str(model.ent_coef) if type(model) == PPO else ""
            cliprange = str(model.clip_range(0)) if type(model) == PPO else ""
            model_name = f"models/{cfg['id']}_simplest_{datetime.datetime.now().strftime('%m%d-%H%M')}"

            log_std_callback = CustomLoggingCallback()

            print(f"logging at {model_name}")
            
            model.learn(total_timesteps=N_TIMESTEPS, 
                        progress_bar=False, 
                        tb_log_name=model_name, 
                        callback=log_std_callback,
                        reset_num_timesteps=False
                        )

            model.save(model_name)

            M,Q,P,B,Z,D = 0, 0, 0, 0, 1, 0
            env = BlendEnv(v = True, 
                        M = M, Q = Q, P = P, B = B, Z = Z, D = D, 
                        action_sample = action_sample, 
                        connections = connections, 
                        tau0 = tau0,
                        delta0 = delta0,
                        sigma = sigma,
                        sigma_ub = sigma_ub,
                        sigma_lb = sigma_lb,
                        s_inv_lb = s_inv_lb,
                        s_inv_ub = s_inv_ub,
                        d_inv_lb = d_inv_lb,
                        d_inv_ub = d_inv_ub,
                        betaT_d = betaT_d,
                        betaT_s = betaT_s,
                        b_inv_ub = b_inv_ub,
                        b_inv_lb = b_inv_lb)
            env = Monitor(env)

            with th.autograd.set_detect_anomaly(True):
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