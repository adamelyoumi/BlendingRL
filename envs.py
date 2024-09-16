import numpy as np
import gymnasium as gym
# from gymnasium import spaces
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from gymnasium.utils import seeding
import json
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import os, sys

"""
    Penalties to add (?):
        -
    Continuous reward: is it really a problem ?
    Are changes only going to be related to supply and demand, or also tank number and connections ? 
        Only supply/demand
"""

def assign_env_config(self, kwargs):
    for key, value in kwargs.items():
        setattr(self, key, value)
    if hasattr(self, 'env_config'):
        for key, value in self.env_config.items():
            # Check types based on default settings
            if hasattr(self, key):
                if type(getattr(self,key)) == np.ndarray:
                    setattr(self, key, value)
                else:
                    setattr(self, key,
                        type(getattr(self, key))(value))
            else:
                raise AttributeError(f"{self} has no attribute, {key}")

def clip(x,a,b):
    if x>b:
        return b
    elif x<a:
        return a
    return x

def dict_to_gym_space(d):
    sub_spaces = {}
    for key, value in d.items():
        if isinstance(value, dict):
            sub_spaces[key] = dict_to_gym_space(value)
        else:
            sub_spaces[key] = Box(low=value, high=value, shape=(1,), dtype=float)
    return Dict(sub_spaces)

def flatten_dict(dictionary, parent_key='', separator=';'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)

def flatten_and_track_mappings(dictionary, separator=';'):
    flattened_dict = flatten_dict(dictionary, separator=separator)
    mappings = [(index, key.split(separator)) for index, (key, value) in enumerate(flattened_dict.items())]
    flattened_array = np.array([value for key, value in flattened_dict.items()]).astype("float32")
    return flattened_array, mappings


def nested_set(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value
    return(dic)
    
def reconstruct_dict(flattened_array, mappings, separator=';'):
    reconstructed_dict = {}
    for index, keys in mappings:
        nested_set(reconstructed_dict, keys, flattened_array[index])
    
    return reconstructed_dict

def get_index(mapping, L):
    for i, items in mapping:
        if items == L:
            return(i)
    raise Exception("Specified list not in mapping")




class BlendEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        """
        
        Args:
            M (int) : Penalty constant incurred for breaking in/out rule. Defaults to 1e3. Set to 0 for "normal" behavior
            Q (int) : Penalty constant incurred for breaking concentrations reqs. Defaults to 1e3. Set to 0 for "normal" behavior
            P (int) : Penalty constant incurred for breaking tank bounds reqs. Defaults to 1e2. Set to 0 for "normal" behavior
            B (int) : Penalty constant incurred for breaking buy/sell bounds reqs. Defaults to 1e2. Set to 0 for "normal" behavior
            Z (int) : Positive reward multiplier to emphasize that "selling is good". Defaults to 1e3. Set to 1 for "normal" behavior
            D (int) : Multiplier representing the influence of the depth. Defaults to 1. Set to 0 for "normal" behavior
            Y (int) : Positive reward multiplier to emphasize that "buying is good". Defaults to 1. Set to 1 for "normal" behavior
            v (bool): Verbose. Defaults to False
            connections (dict) : Specifies connection graph and tank names
            action_samples (dict) : Action example for action space definition
        
        """
        super().__init__()
        
        self.T = 6
        self.alpha = 0.1
        self.beta = 0
        self.v = False # Verbose
        
        self.M = 1e3            # Negative reward (penalty) constant factor for breaking in/out rule
        self.Q = 1e3            # Negative reward (penalty) constant factor for breaking concentrations reqs
        self.P = 1e2            # Negative reward (penalty) constant factor for breaking tank bounds reqs
        self.B = 1e2            # Negative reward (penalty) constant factor for breaking buy/sell bouds reqs
        self.Z = 1e3            # Positive reward multiplier to emphasize that "selling is good"
        self.D = 1              # Multiplier representing the influence of the depth
        self.Y = 1              # Positive reward multiplier to emphasize that "buying is good"
        
        self.eps = 1e-2         # Tolerance for breaking in/out rule and other "== 0" checks
        self.reg = 0            # Regularization type. 0 for no reg, 1 for L1 reg., 2 for L2 etc
        self.reg_lambda = 1     # Regularization factor
        self.MAXFLOW = 50
        self.determ = True
        
        with open("./configs/json/connections_base.json" ,"r") as f:
            connections_s = f.readline()
        self.connections = json.loads(connections_s)
        
        self.properties = ["q1"]
        
        self.tau0   = {'s1': [10, 10, 10, 0, 0, 0],  's2': [30, 30, 30, 0, 0, 0]}
        self.delta0 = {'p1': [0, 0, 15, 15, 15, 15], 'p2': [0, 0, 15, 15, 15, 15]}
        
        self.sigma = {"s1":{"q1": 0.06}, "s2":{"q1": 0.26}} # Source concentrations
        self.sigma_ub = {"p1":{"q1": 0.16}, "p2":{"q1": 1}} # Demand concentrations UBs/LBs
        self.sigma_lb = {"p1":{"q1": 0}, "p2":{"q1": 0}}
        
        self.s_inv_lb = {'s1': 0, 's2': 0}
        self.s_inv_ub = {'s1': 999, 's2': 999}
        self.d_inv_lb = {'p1': 0, 'p2': 0}
        self.d_inv_ub = {'p1': 999, 'p2': 999}
        
        self.betaT_d = {'p1': 2, 'p2': 1} # Price of sold products
        self.betaT_s = {'s1': 0, 's2': 0} # Cost of bought products
        
        self.b_inv_ub = {"j1": 30, "j2": 30, "j3": 30, "j4": 30, "j5": 20, "j6": 20, "j7": 20, "j8": 20} 
        self.b_inv_lb = {j:0 for j in self.b_inv_ub.keys()} 
        
        self.forecast_window_len = 6
        
        with open("./configs/json/action_sample_base.json" ,"r") as f:
            action = f.readline()
        self.action_sample = json.loads(action)
        
        
        assign_env_config(self, kwargs)
        
        
        self.sources = list(self.tau0.keys())
        self.demands = list(self.delta0.keys())
        self.blenders = list(self.connections["blend_blend"].keys())
        
        for s in self.sources:
            self.tau0[s].append(0)
        for p in self.demands:
            self.delta0[p].append(0)
        
        
        self.depths = {"s1": self.D*1, "s2": self.D*1,
                       "j1": self.D*2, "j2": self.D*2, "j3": self.D*2, "j4": self.D*2, 
                       "j5": self.D*3, "j6": self.D*3, "j7": self.D*3, "j8": self.D*3,
                       "p1": self.D*4, "p2": self.D*4}
        
        self.reset() # sets state, reward, t, done
        
        # print("sss", self.state)
        
        self.flatt_state, self.mapping_obs = flatten_and_track_mappings(self.state)
        
        # print(self.mapping_obs)
        self.observation_space = Box(low=0, high=self.MAXFLOW, shape=(self.flatt_state.shape[0],))
        
        
        self.flatt_act_sample, self.mapping_act = flatten_and_track_mappings(self.action_sample)
        # self.logg(f"action space shape: {len(self.flatt_act_sample)}")
        self.action_space = Box(low=0, high=self.MAXFLOW, shape=(len(self.flatt_act_sample),))
        # self.action_space = Dict({"a": Box(low=0, high=self.MAXFLOW, shape=(len(self.flatt_act_sample),))})
        
    def step(self, action: th.Tensor):
        """
        The state is kept track of in a human-readable dict "self.state"
        After updating it from the action, we flatten it and return it along with the reward and "done"

        Args:
            action (torch.Tensor): model output
        """
        
        self.t += 1
        if self.t == self.T:
            self.done = True
        
        # Applying regularization before adjusting action
        
        regularization_term = self.reg_lambda * (th.norm(th.Tensor(action), p=self.reg).item() if self.reg else 0)
        self.reward -= regularization_term
        self.pen_tracker["reg"] -= regularization_term
        
        action = action.tolist()
        action = reconstruct_dict(action, self.mapping_act) # From non-human-readable list to human-readable dict
        action = self.sanitize_action_structure(action) # Modify action according to chosen rules (ex: set flows < 0.1 to 0 etc)
        
        action = self.penalize_action_preflows(action)
        
        self.update_reward1(action)
        
        prev_blend_invs = self.state["blenders"]
        
        
        for s in self.sources:
            
            # I + t - (x+y) > M: I + at - (x+y) = M => a = (M+(x+y)-I)/t
            # I + t - (x+y) < m: I + t - b(x+y) = m => b = (I+t-m)/(x+y)
            
            outgoing = sum([action["source_blend"][s][j] for j in action["source_blend"][s].keys()])
            newinv = self.state["sources"][s] - outgoing + action["tau"][s]
            # Enforcing bounds
            if newinv > self.s_inv_ub[s]: # inv too high -> reduce bought amount
                self.logg(f"{s}: newtau: {self.s_inv_ub[s] + outgoing - self.state['sources'][s]}")
                action["tau"][s] = self.s_inv_ub[s] + outgoing - self.state["sources"][s]
                
                self.logg(f"[PEN] t{self.t}; {s}:\t\t\tbought too much (resulting amount more than source tank UB)")
                self.reward -= self.B
                self.pen_tracker["B"] -= self.B
                self.violated_pens += 1
                
            elif newinv < self.s_inv_lb[s]: # inv too low -> reduce outgoing amount
                b = (self.state["sources"][s] + action["tau"][s] - self.s_inv_lb[s])/outgoing
                self.logg(f"{s}: b: {b}")
                for j in action["source_blend"][s].keys():
                    action["source_blend"][s][j] *= b
                
                self.logg(f"[PEN] t{self.t}; {s}:\t\t\tbought too little (resulting amount less than source tank LB)")
                self.reward -= self.B
                self.pen_tracker["B"] -= self.B
                self.violated_pens += 1

            # Giving reward depending on depths
            
            incr = self.depths[s] * max(0, newinv - self.state["sources"][s])
            if self.D:
                self.logg(f"Increased reward by {incr} through tank population in {s}")
            self.reward += incr
            
            # Updating inv
            newinv = self.state["sources"][s] - sum([action["source_blend"][s][j] for j in action["source_blend"][s].keys()]) + action["tau"][s]
            self.state["sources"][s] = clip(newinv, self.s_inv_lb[s], self.s_inv_ub[s])
        
        
        # self.logg("Action after processing sources:", action)
        
        for j in self.blenders:
            # Computing inflow and outflow
            in_flow_sources = in_flow_blend = out_flow_blend = out_flow_demands = 0
            for s in self.sources:
                if j in action["source_blend"][s].keys():
                    in_flow_sources += action["source_blend"][s][j]
            for jp in self.blenders:
                if "blend_blend" in action.keys() and j in action["blend_blend"][jp].keys():
                    in_flow_blend += action["blend_blend"][jp][j]
                if "blend_blend" in action.keys() and jp in action["blend_blend"][j].keys():
                    out_flow_blend += action["blend_blend"][j][jp]
            for p in self.demands:
                if p in action["blend_demand"][j].keys():
                    out_flow_demands += action["blend_demand"][j][p]
            
            self.logg(f"{j}: inv: {self.state['blenders'][j]}, in_flow_sources: {in_flow_sources}, in_flow_blend: {in_flow_blend}, out_flow_blend: {out_flow_blend}, out_flow_demands: {out_flow_demands}")
            
            # Enforcing No in and out flow
            if (in_flow_sources + in_flow_blend > self.eps) and (out_flow_blend + out_flow_demands > self.eps):
                self.logg(f"[PEN] t{self.t}; {j}:\t\t\tIn and out flow both non-zero (in: {round(in_flow_sources + in_flow_blend, 2)}, out: {round(out_flow_blend + out_flow_demands, 2)})")
                self.reward -= self.M
                self.pen_tracker["M"] -= self.M
                self.violated_pens += 1
                
                # Choice: we remove all flows. We can also remove only outgoing flows, only incoming flows, or decide based on the tank's position
                # (if the tank is connected to sources, then keep incoming flow, but if it is connected to demands, then keep outgoing flow)
                for s in self.sources:
                    if j in action["source_blend"][s].keys():
                        action["source_blend"][s][j] = 0
                for jp in self.blenders:
                    if "blend_blend" in action.keys() and j in action["blend_blend"][jp].keys():
                        action["blend_blend"][jp][j] = 0
                    if "blend_blend" in action.keys() and jp in action["blend_blend"][j].keys():
                        action["blend_blend"][j][jp] = 0
                for p in self.demands:
                    if p in action["blend_demand"][j].keys():
                        action["blend_demand"][j][p] = 0
                
                continue # Inventory does not change
            
            # else...
            in_flow_sources = max(0, in_flow_sources)
            in_flow_blend = max(0, in_flow_blend)
            out_flow_blend = max(0, out_flow_blend)
            out_flow_demands = max(0, out_flow_demands)
            newinv = self.state["blenders"][j] + in_flow_sources + in_flow_blend - out_flow_blend - out_flow_demands
            
            # I + w+x-(y+z) > M : I + a(w+x) - (y+z) = M  =>  a = (M+y+z-I)/(w+x)
            # I + w+x-(y+z) < m : I + (w+x) - b(y+z) = m  =>  b = (I+w+x-m)/(y+z)
            
            # Enforcing inventory bounds
            # NB: we assume "no in and out" rule is respected
            if newinv > self.b_inv_ub[j]: # inv too high -> reduce incoming amount
                self.logg(action)
                a = (self.b_inv_ub[j] + out_flow_blend + out_flow_demands - self.state["blenders"][j])/(in_flow_sources + in_flow_blend)
                self.logg(f"{j}: a: {a}")
                for s in self.sources:
                    if j in action["source_blend"][s].keys():
                        action["source_blend"][s][j] *= a
                for jp in self.blenders:
                    if "blend_blend" in action.keys() and j in action["blend_blend"][jp].keys():
                        action["blend_blend"][jp][j] *= a
                
                self.logg(f"[PEN] t{self.t}; {j}:\t\t\tinventory OOB (resulting amount more than blending tank UB)")
                self.reward -= self.P
                self.pen_tracker["P"] -= self.P
                self.violated_pens += 1
                
            elif newinv < self.b_inv_lb[j]: # inv too low -> reduce outgoing amount
                b = (self.state["blenders"][j] + in_flow_sources + in_flow_blend - self.b_inv_lb[j])/(out_flow_blend + out_flow_demands)
                self.logg(f"{j}: b: {b}")
                for jp in self.blenders:
                    if "blend_blend" in action.keys() and jp in action["blend_blend"][j].keys():
                        action["blend_blend"][j][jp] *= b
                for p in self.demands:
                    if p in action["blend_demand"][j].keys():
                        action["blend_demand"][j][p] *= b
                        
                self.logg(f"[PEN] t{self.t}; {j}:\t\t\tinventory OOB (resulting amount less than blending tank LB)")
                self.reward -= self.P
                self.pen_tracker["P"] -= self.P
                self.violated_pens += 1
            
            incr = self.depths[j] * max(0, newinv - self.state["blenders"][j])
            if self.D:
                self.logg(f"Increased reward by {incr} through tank population in {j}")
            self.reward += incr
            
            # Computing rectified newinv
            in_flow_sources = in_flow_blend = out_flow_blend = out_flow_demands = 0
            for s in self.sources:
                if j in action["source_blend"][s].keys():
                    in_flow_sources += action["source_blend"][s][j]
            for jp in self.blenders:
                if "blend_blend" in action.keys() and j in action["blend_blend"][jp].keys():
                    in_flow_blend += action["blend_blend"][jp][j]
                if "blend_blend" in action.keys() and jp in action["blend_blend"][j].keys():
                    out_flow_blend += action["blend_blend"][j][jp]
            for p in self.demands:
                if p in action["blend_demand"][j].keys():
                    out_flow_demands += action["blend_demand"][j][p]
                    
            newinv = self.state["blenders"][j] + in_flow_sources + in_flow_blend - out_flow_blend - out_flow_demands
            self.state["blenders"][j] = clip(newinv, self.b_inv_lb[j], self.b_inv_ub[j])
            
        
        # self.logg("Action after processing blenders:", action)
        
        for p in self.demands:
            
            # I + (x+y)-d > M: I + a(x+y) - d = M  =>  a = (M+d-I)/(x+y)
            # I + (x+y)-d < m: I + (x+y) - bd = m  =>  b = (I+(x+y)-m)/d
            
            incoming = 0
            for jp in self.blenders:
                if p in action["blend_demand"][jp].keys():
                    incoming += action["blend_demand"][jp][p]
            
            newinv = self.state["demands"][p] + incoming - action["delta"][p] 
            
            # Enforcing inventory bounds
            if newinv > self.d_inv_ub[p]: # inv too high -> reduce incoming amount
                a = (self.d_inv_ub[p] + action["delta"][p] - self.state["demands"][p])/incoming
                self.logg(f"{p}: a: {a}")
                
                for jp in self.blenders:
                    if p in action["blend_demand"][jp].keys():
                        action["blend_demand"][jp][p] *= a
                        
                self.reward -= self.B
                self.pen_tracker["B"] -= self.B
                self.logg(f"[PEN] t{self.t}; {p}:\t\t\tsold too little (resulting amount more than demand tank UB)")
                self.violated_pens += 1
                
            elif newinv < self.d_inv_lb[p]:  # inv too low -> reduce sold amount
                self.logg(f"{p}: newdelta: {self.state['demands'][p] + incoming - self.d_inv_lb[p]}")
                action["delta"][p] = self.state["demands"][p] + incoming - self.d_inv_lb[p]
                
                self.reward -= self.B
                self.pen_tracker["B"] -= self.B
                self.logg(f"[PEN] t{self.t}; {p}:\t\t\tsold too much (resulting amount less than demand tank LB)")
                self.violated_pens += 1
            
            
            incr = self.depths[p] * max(0, newinv - self.state["demands"][p])
            if self.D:
                self.logg(f"Increased reward by {incr} through tank population in {p}")
            self.reward += incr
            
            incoming = 0
            for jp in self.blenders:
                if p in action["blend_demand"][jp].keys():
                    incoming += action["blend_demand"][jp][p]
            
            # self.logg("xxx", newinv, clip(newinv, self.d_inv_lb[p], self.d_inv_ub[p]), self.state["demands"][p], incoming, action["delta"][p], self.state["blenders"]["j1"])
            
            newinv = self.state["demands"][p] + incoming - action["delta"][p] 
            self.state["demands"][p] = clip(newinv, self.d_inv_lb[p], self.d_inv_ub[p])
            
            # self.logg(self.state)
        
        # self.logg("Action after processing demands:", action)
        
        # Properties                      
        for j in self.blenders:
            for q in self.properties:
                if self.state["blenders"][j] < self.eps:
                    self.state["properties"][j][q] = 0
                else:
                    in_flow_sources = in_flow_blend = out_flow_blend = out_flow_demands = 0
                    for s in self.sources:
                        if j in action["source_blend"][s].keys():
                            in_flow_sources += action["source_blend"][s][j] * self.sigma[s][q]
                    for jp in self.blenders:
                        if "blend_blend" in action.keys() and j in action["blend_blend"][jp].keys():
                            in_flow_blend += action["blend_blend"][jp][j] * self.state["properties"][jp][q]
                        if "blend_blend" in action.keys() and jp in action["blend_blend"][j].keys():
                            out_flow_blend += action["blend_blend"][j][jp] * self.state["properties"][jp][q]
                    for p in self.demands:
                        if p in action["blend_demand"][j].keys():
                            out_flow_demands += action["blend_demand"][j][p] * self.state["properties"][j][q]

                    self.state["properties"][j][q] = (1/self.state["blenders"][j]) * ( \
                                                    self.state["properties"][j][q] * prev_blend_invs[j] \
                                                    + in_flow_sources + in_flow_blend - out_flow_blend - out_flow_demands
                                                )

        
        self.update_reward2(action)
        
        for s in self.sources:
            for k in range(self.forecast_window_len):
                self.state[f"sources_avail_next_{k}"][s] = self.tau0[s][k + self.t] if k + self.t < len(self.tau0[s]) else 0
        
        for p in self.demands:
            for k in range(self.forecast_window_len):
                self.state[f"demands_avail_next_{k}"][p] = self.delta0[p][k + self.t] if k + self.t < len(self.delta0[p]) else 0
        
        self.state["t"] = self.t
        
        self.flatt_state, _ = flatten_and_track_mappings(self.state)
        
        
        ### Commented out to see if it is needed or not (shouldn't be)
        # for k in range(self.flatt_state.shape[0]):
        #     self.flatt_state[k] = max(0, self.flatt_state[k]) 
        
        
        return self.flatt_state, self.reward, self.done, False, {"dict_state": self.state, "pen_tracker": self.pen_tracker}
    
    def get_new_start_state_deterministic(self):
        
        self.state = {
            "sources": {s:0 for s in self.sources},
            "blenders": {b:0 for b in self.blenders},
            "demands": {p:0 for p in self.demands},
            "properties": {b: {q:0 for q in self.properties} for b in self.blenders}
        }
        
        for k in range(self.forecast_window_len):
            # self.state[f"sources_avail_next_{k}"] = {s: self.tau0[s][k]   if k < self.T else 0 for s in self.sources}
            # self.state[f"demands_avail_next_{k}"] = {p: self.delta0[p][k] if k < self.T else 0 for p in self.demands}
            self.state[f"sources_avail_next_{k}"] = {s: self.tau0[s][k]   if k < len(self.tau0[s]) else 0 for s in self.sources}
            self.state[f"demands_avail_next_{k}"] = {p: self.delta0[p][k] if k < len(self.delta0[p]) else 0 for p in self.demands}
            
        self.state["t"] = self.t
    
    def get_new_start_state_probabilistic(self, seed):
        
        np.random.seed(seed)
        
        self.tau0   = {'s1': [np.random.randint(10,30), np.random.randint(10,30), np.random.randint(10,30), 0, 0, 0], 
                       's2': [np.random.randint(10,30), np.random.randint(10,30), np.random.randint(10,30), 0, 0, 0]}
        available_chems = sum(self.tau0["s1"]) + sum(self.tau0["s2"])
        
        self.delta0 = {'p1': [0, 0, 0, available_chems/6, available_chems/6, available_chems/6], 
                       'p2': [0, 0, 0, available_chems/6, available_chems/6, available_chems/6]}
        
        self.state = {
            "sources": {s:0 for s in self.sources},
            "blenders": {b:0 for b in self.blenders},
            "demands": {p:0 for p in self.demands},
            "properties": {b: {q:0 for q in self.properties} for b in self.blenders},
            # How much we can buy at the current timestamp. Noted as "F^{IN}_{s,t}" in the paper
            "sources_avail_next": {s: self.tau0[s][:self.forecast_window_len]   for s in self.sources},
            # How much we can sell at the current timestamp. Noted as "FD^L_{p,t}" in the paper
            "demands_avail_next": {p: self.delta0[p][:self.forecast_window_len] for p in self.demands}  
        }

    def update_reward1(self, action):
        """
            Follows the definition/structure of the Overleaf Document

        Args:
            action (dict): See action_sample.json .
        """
        Q_float = Q_bin = 0
        if "blend_blend" in action.keys():
            L = ["source_blend", "blend_blend", "blend_demand"]
        else:
            L = ["source_blend", "blend_demand"]
        for k in L:
            for tank1 in action[k].keys():
                for tank2 in action[k][tank1].keys():
                    Q_float += action[k][tank1][tank2]
                    Q_bin   += 1 if action[k][tank1][tank2] > 0 else 0 
                    
        self.reward -= (self.alpha * Q_bin + self.beta * Q_float)
        
    def update_reward2(self, action):
        R2 = 0
        
        for p in self.demands:
            R2 += self.betaT_d[p] * action["delta"][p] * self.Z
        for s in self.sources:
            R2 -= self.betaT_s[s] * action["tau"][s] * self.Y
            
        for j in self.blenders:
            R2 -= self.penalty_in_out_flow(j, action)
            for q in self.properties:
                for p in self.demands:
                    R2 -= self.penalty_quality(p, q, j, action)

        self.reward += R2
        
    def penalty_quality(self, p, q, j, action):
        if (self.state["properties"][j][q] < self.sigma_lb[p][q] or self.state["properties"][j][q] > self.sigma_ub[p][q]) \
                and (p in action["blend_demand"][j].keys() and action["blend_demand"][j][p] > 0):
            self.logg(f"[PEN] t{self.t}; {p}; {q}; {j}:\t\t\tSold qualities out of bounds ({round(self.state['properties'][j][q], 2)})")
            self.violated_pens += 1
            return self.Q
        return 0
    
    def penalty_in_out_flow(self, j, action):
        sum_in = sum_out = 0
        if "blend_blend" in action.keys():
            for jp in self.blenders:
                sum_in  += action["blend_blend"][jp][j] if j in action["blend_blend"][jp].keys() else 0
                sum_out += action["blend_blend"][j][jp] if jp in action["blend_blend"][j].keys() else 0
        
        for s in self.sources:
            sum_in  += action["source_blend"][s][j] if j in action["source_blend"][s].keys() else 0
        
        for p in self.demands:
            sum_out += action["blend_demand"][j][p] if p in action["blend_demand"][j].keys() else 0
            
        if sum_in > self.eps and sum_out > self.eps: # /!\
            self.logg(f"[PEN] t{self.t}; {j}:\t\t\tIn and out flow both non-zero (in: {round(sum_in, 2)}, out:{round(sum_out, 2)})")
            self.violated_pens += 1
            return self.M
        
        return 0
    
    def sanitize_action_structure(self, action):
        """Normalize model action if needed
        
        Incur penalties if the action leads to impossible/unauthorized states

        Args:
            action (dict): Action dict
        """
        if "blend_blend" not in action.keys():
            return(action)
        
        for j in self.blenders:
            if j not in action["blend_blend"].keys():
                action["blend_blend"][j] = {}
            if j not in action["blend_demand"].keys():
                action["blend_demand"][j] = {}
        return(action)
        
    def penalize_action_preflows(self, action, pen = True):
        """Add Penalty if the action is illegal (before flows are processed).
        Includes penalties related to the model proposing to buy/sell more product than the demands/sources allow (not inventory).

        Args:
            action (dict)
            pen (bool, optional): Set to False to disable penalties. Defaults to True.
        """
        
        # Add penalty and log if trying to buy too much product
        for s in self.sources:
            if action["tau"][s] > self.state["sources_avail_next_0"][s]:
                action["tau"][s] = self.state["sources_avail_next_0"][s]
                self.reward -= self.B if pen else 0 # incur penalty
                self.pen_tracker["B"] -= self.B
                self.logg(f"[PEN] t{self.t}; {s}:\t\t\tbought too much (more than supply)")
                self.violated_pens += 1
        
        # Add penalty and log if trying to sell too much product (more than available demand or more than available inventory)
        for p in self.demands:
            if action["delta"][p] > self.state["demands_avail_next_0"][p]:
                action["delta"][p] = self.state["demands_avail_next_0"][p]
                self.reward -= self.B if pen else 0 # incur penalty
                self.pen_tracker["B"] -= self.B
                self.logg(f"[PEN] t{self.t}; {p}:\t\t\tsold too much (more than demand)")
                self.violated_pens += 1
        
        return action
    
    def penalize_action_postflows(self, action, pen = True):
        ########## UNUSED ##########
        """Add Penalty if the action is illegal (after flows are processed but BEFORE product is sold/bought)
        Includes penalties related to the model proposing to buy/sell more product than the demands/sources allow (inventory), 
        Fix illegal state accordingly

        Args:
            action (dict)
            pen (bool, optional): Set to False to disable penalties. Defaults to True.
        """
        for s in self.sources:
            if self.state["sources"][s] >= self.s_inv_ub[s] or self.state["sources"][s] <= self.s_inv_lb[s]:
                self.state["sources"][s] = clip(self.state["sources"][s], self.s_inv_lb[s], self.s_inv_ub[s])
                self.reward -= self.P if pen else 0 # incur penalty
                self.logg(f"[PEN] t{self.t}; {s}:\t\t\tinventory out of bounds")
                self.violated_pens += 1
                
            if action["tau"][s] > self.s_inv_ub[s] - self.state["sources"][s]:
                action["tau"][s] = self.state["sources"][s]
                self.reward -= self.B if pen else 0 # incur penalty
                self.pen_tracker["B"] -= self.B
                self.logg(f"[PEN] t{self.t}; {s}:\t\t\tbought too much (resulting amount more than source tank UB)")
                self.violated_pens += 1
                
        for j in self.blenders:
            if self.state["blenders"][j] >= self.b_inv_ub[j] or self.state["blenders"][j] <= self.b_inv_lb[j]:
                self.state["blenders"][j] = clip(self.state["blenders"][j], self.b_inv_lb[j], self.b_inv_ub[j])
                self.reward -= self.P if pen else 0 # incur penalty
                self.logg(f"[PEN] t{self.t}; {j}:\t\t\tinventory out of bounds")
                self.violated_pens += 1
        
        for p in self.demands:
            if self.state["demands"][p] >= self.d_inv_ub[p] or self.state["demands"][p] <= self.d_inv_lb[p]:
                self.state["demands"][p] = clip(self.state["demands"][p], self.d_inv_lb[p], self.d_inv_ub[p])
                self.reward -= self.P if pen else 0 # incur penalty
                self.logg(f"[PEN] t{self.t}; {p}:\t\t\tinventory out of bounds")
                self.violated_pens += 1
                
            if action["delta"][p] > self.state["demands"][p] - self.d_inv_lb[p]:
                action["delta"][p] = self.state["demands"][p]
                self.reward -= self.B if pen else 0 # incur penalty
                self.pen_tracker["B"] -= self.B
                self.logg(f"[PEN] t{self.t}; {p}:\t\t\tsold too much (resulting amount less than tank LB)")
                self.violated_pens += 1
                
                # tau = 50
                # state = 20
                # ub = 60
                # tau > ub-state
    
    def reset(self, seed=0):
        self.violated_pens = 0
        self.t = 0
        
        if self.determ:
            self.get_new_start_state_deterministic()
        else:
            self.get_new_start_state_probabilistic(seed)
            
        self.reward = 0
        self.done = False
        
        self.pen_tracker = {"M": 0, "B": 0, "P": 0, "Q": 0, "reg": 0}
        
        self.flatt_state, _ = flatten_and_track_mappings(self.state)
        return self.flatt_state, {"dict_state": self.state, "pen_tracker": self.pen_tracker}
        
    def render(self, action = None):
        
        print(f"\nt{self.t}:\n")
        
        if action is not None:
            M = [[action["source_blend"][s][b] for b in self.blenders] for s in self.sources]
            N = [[action["blend_blend"][b1][b2] for b2 in self.blenders] for b1 in self.blenders]
            O = [[action["blend_demand"][b][p] for p in self.demands] for b in self.blenders]

        Mi = [self.state["sources"][s] for s in self.sources]
        Ni = [self.state["blenders"][b] for b in self.blenders]
        Oi = [self.state["demands"][p] for p in self.demands]

        print("\n\n\n##################### ACTION FLOW VARIABLES #####################\n\n")
        
        print("s2b")
        for s in range(len(self.sources)):
            print(M[s])
        print("b2b")
        for b in range(len(self.blenders)):
            print(N[b])
        print("b2p")
        for b in range(len(self.blenders)):
            print(O[b])

        print("\n\n\n##################### INVENTORY VARIABLES #####################\n\n")
        print(Mi)
        print(Ni)
        print(Oi)
            
        print("\n\n\n\nReward:", self.reward)
        
    def logg(self, *args):
        if self.v:
            print(*args)
        return

        
class MaskedBlendEnv(BlendEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.action_space = MultiDiscrete(len(self.flatt_act_sample)*[int((self.MAXFLOW/5)+1)])
        
        self.act_mask      = th.ones(self.action_space.shape[0], dtype=th.float32)
        self.act_mask_dict = reconstruct_dict(self.act_mask, self.mapping_act)
        
    def step(self, action):
        action = action * 5
        obs, reward, done, term, info = super().step(action)
        
        self.action_masks()
        
        return obs, reward, done, term, info
        
    def action_masks(self):
        self.act_mask      = th.tensor([1] * self.action_space.shape[0], dtype=th.float32)
        self.act_mask_dict = reconstruct_dict(self.act_mask, self.mapping_act)
        
        for s in self.sources:
            if self.state["sources"][s] == self.s_inv_lb:  # No outgoing flow
                for jp in self.act_mask_dict["source_blend"][s].keys():
                    self.act_mask_dict["source_blend"][s][jp] = 0
                    
            if self.state["sources"][s] == self.s_inv_ub or self.state["sources_avail_next_0"] == 0:  # No incoming flow
                self.act_mask_dict["tau"][s] = 0
            
                
        for j in self.blenders:
            if self.state["blenders"][j] == self.b_inv_lb:  # No outgoing flow
                if j in self.act_mask_dict["blend_blend"].keys():
                    for jp in self.act_mask_dict["blend_blend"][j].keys():
                        self.act_mask_dict["blend_blend"][j][jp] = 0
                if j in self.act_mask_dict["blend_demand"].keys():
                    for p in self.act_mask_dict["blend_demand"][j].keys():
                        self.act_mask_dict["blend_demand"][j][p] = 0
            
            if self.state["blenders"][j] == self.b_inv_ub:  # No incoming flow
                for jp in self.act_mask_dict["blend_blend"].keys():
                    if j in self.act_mask_dict["blend_blend"][jp].keys():
                        self.act_mask_dict["blend_blend"][jp][j] = 0
                for s in self.act_mask_dict["source_blend"].keys():
                    self.act_mask_dict["source_blend"][s][j] = 0
                
                
        for p in self.demands:
            if self.state["demands"][p] == self.d_inv_lb or self.state["demands_avail_next_0"] == 0: # No outgoing flow
                self.act_mask_dict["delta"][p] = 0
                
            if self.state["demands"][p] == self.d_inv_ub:  # No incoming flow
                for jp in self.act_mask_dict["blend_demand"].keys():
                    self.act_mask_dict["blend_demand"][jp][p] = 0
                    
        self.act_mask, _ = flatten_and_track_mappings(self.act_mask_dict)
        
        L = []
        for k in self.act_mask:
            L.append([k]*int((self.MAXFLOW/5)+1))
            L[-1][0] = 1
        
        return np.array(L, dtype=np.float32)


tau0   = {'s1': [10, 10, 10, 0, 0, 0]}
delta0 = {'p1': [0, 0, 0, 10, 10, 10]}
sigma = {"s1":{"q1": 0.06}} # Source concentrations
sigma_ub = {"p1":{"q1": 0.16}} # Demand concentrations UBs/LBs
sigma_lb = {"p1":{"q1": 0}}
s_inv_lb = {'s1': 0}
s_inv_ub = {'s1': 999}
d_inv_lb = {'p1': 0}
d_inv_ub = {'p1': 999}
betaT_d = {'p1': 1} # Price of sold products
betaT_s = {'s1': 0} # Cost of bought products
b_inv_ub = {"j1": 30} 
b_inv_lb = {j:0 for j in b_inv_ub.keys()} 

connections = {
    "source_blend": {"s1": ["j1"]},
    "blend_blend": {"j1": []},
    "blend_demand": {"j1": ["p1"]}
}

action_sample = {
    'source_blend':{'s1': {'j1':1}},
    'blend_blend':{},
    'blend_demand':{'j1': {'p1':1}},
    "tau": {"s1": 10},
    "delta": {"p1": 0}
}

simplestenv = BlendEnv(v = False, 
               D=0, Q = 0, P = 0, B = 0, Z = 1E4, M = 1E3,
               connections = connections, 
               action_sample = action_sample,
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