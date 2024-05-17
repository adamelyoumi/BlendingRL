import numpy as np
import gymnasium as gym
# from gymnasium import spaces
from gymnasium.spaces import Box, Dict
from gymnasium.utils import seeding
from or_gym.utils import assign_env_config
import json
import torch as t

"""
    How to convert state dict to model input, and model output back to action dict ?
    Use of a class-based environment ?
    Penalties to add (?):
        - Flow > Inventory (or let the model learn by itself and we cap the action before applying it ?)
        - Sell > demand
        - Buy > Supply
    Continuous reward: is it really a problem ?
    Are changes only going to be related to supply and demand, or also tank number and connections ?
"""

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



class PoolEnv(gym.Env):
    
    def __init__(self, *args, **kwargs):
        
        """
        
            Implementation of the Pooling Problem Environment
            See page 41 of https://optimization-online.org/wp-content/uploads/2015/04/4864.pdf
            
            Flow connection graph (adjacencies)                                                   : fixed for a given Env instance.
            Costs, Initial concentrations, prices, concentration requirements, max requirement    : in observation (randomized). 1 episode = 1 step
            F_{i,j}, F_X, F_Y                                                                     : to be provided by the model (the action)
            Reward                                                                                : Calculated from costs & prices
            
        """
        
        self.N_pool = 1
        self.N_supply = 3
        self.N_demand = 2
        
        self.s2t = {("s3", "t1"),
                    ("s3", "t2")}
        
        self.s2p = {("s1", "p1"),
                    ("s2", "p1")}
        
        self.p2t = {("p1", "t1"),
                    ("p1", "t2")}
        
        ### TODO ###
        
        assign_env_config(self, kwargs)
        self.set_seed()
        
        self.N = self.N_blending + self.N_supply + self.N_demand

        obs_space = ...
        
        self.action_space = ...
        
        if self.mask:
            self.observation_space = Dict({
                "action_mask": ...,
                "avail_actions": ...,
                "state": obs_space
                })
        else:
            self.observation_space = ...
        
        self.reset()
        
    
    def sample_action(self):
        return

    def reset(self):
        return

    def step(self, action):
        return
        
    def render(self):
        return True

    
    def set_seed(self, seed=None):
        if seed == None:
            seed = np.random.randint(0, np.iinfo(np.int32).max)        
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

class BlendEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.T = 6
        self.alpha = 0.1
        self.beta = 0
        self.v = False # Verbose
        
        self.logg = lambda *args: print(*args) if self.v else lambda *args: None
        
        self.M = 1e2 # Negative reward (penalty) constant factor
        self.P = 1e1 # Negative reward (penalty) constant factor (small)
        self.Z = 1e4 # Positive reward multiplier to emphasize that "selling is good"
        
        self.MAXFLOW = 150
        self.determ = True
        
        with open("./connections_sample.json" ,"r") as f:
            connections_s = f.readline()
        self.connections = json.loads(connections_s)
        
        self.properties = ["q1"]
        
        self.tau0   = {'s1': [10, 10, 10, 0, 0, 0],  's2': [30, 30, 30, 0, 0, 0]}
        self.delta0 = {'p1': [0, 0, 15, 15, 15, 15], 'p2': [0, 0, 15, 15, 15, 15]}
        
        self.sigma = {"s1":{"q1": 0.06}, "s2":{"q1": 0.26}} # Source concentrations
        self.sigma_ub = {"p1":{"q1": 0.16}, "p2":{"q1": 1}} # Demand concentrations UBs/LBs
        self.sigma_lb = {"p1":{"q1": 0}, "p2":{"q1": 0}}
        
        self.s_inv_lb = {'s1': 0, 's2': 0}          # Unused
        self.s_inv_ub = {'s1': 0, 's2': 0}          # Unused
        self.d_quals_lb = {'p1': 0, 'p2': 0}        # Unused
        self.d_quals_ub = {'p1': 0.16, 'p2': 0.1}   # Unused
        self.d_inv_lb = {'p1': 0, 'p2': 0}          # Unused
        self.d_inv_ub = {'p1': 0, 'p2': 0}          # Unused
        
        self.betaT_d = {'p1': 2, 'p2': 1} # Price of sold products
        self.betaT_s = {'s1': 0, 's2': 0} # Cost of bought products
        
        self.b_inv_ub = {"j1": 30, "j2": 30, "j3": 30, "j4": 30, "j5": 20, "j6": 20, "j7": 20, "j8": 20} # Unused
        
        self.sources = list(self.tau0.keys())
        self.demands = list(self.delta0.keys())
        self.blenders = list(self.connections["blend_blend"].keys())
        
        assign_env_config(self, kwargs)
        
        
        self.reset() # sets state, reward, t, done
        
        self.flatt_state, self.mapping_obs = flatten_and_track_mappings(self.state)
        self.observation_space = Box(low=0, high=self.MAXFLOW, shape=(self.flatt_state.shape[0],))
        
        with open("./action_sample.json" ,"r") as f:
            action = f.readline()
        action = json.loads(action)
        
        self.logg("ACTION:",action)
        
        
        # Do not sanitize structure for smaller action space ?
        
        # self.logg("action before sanitizing:", action)
        # action = self.sanitize_action_structure(action, pen = False) # sanitizing to avoid format inconsistencies
        # self.logg("action after sanitizing:",action)
        
        self.flatt_act_sample, self.mapping_act = flatten_and_track_mappings(action)
        self.logg(f"action space shape: {len(self.flatt_act_sample)}")
        self.action_space = Box(low=0, high=self.MAXFLOW, shape=(len(self.flatt_act_sample),))
        
        
    def step(self, action: t.Tensor):
        """
        The state is kept track of in a human-readable dict "self.state"
        After updating it from the action, we flatten it and return it along with the reward and "done"

        Args:
            action (torch.Tensor): model output
        """
        """
        Questions:
            How to convert state dict to model input, and model output back to action dict ?
            Use of a class-based environment ?
        """
        """
        action_sample = {
            'source_blend':{
                's1': {'j1':0, 'j2':0, 'j3':0, 'j4':0}, # From s1 to b1, from s1 to b2 etc...
                's2': {'j1':0, 'j2':0, 'j3':0, 'j4':0},
            },
            
            'blend_blend':{
                'j1': {'j5':0, 'j6':0, 'j7':0, 'j8':0},
                'j2': {'j5':0, 'j6':0, 'j7':0, 'j8':0},
                'j3': {'j5':0, 'j6':0, 'j7':0, 'j8':0},
                'j4': {'j5':0, 'j6':0, 'j7':0, 'j8':0},
                'j5': {},
                'j6': {},
                'j7': {},
                'j8': {}
            },
            
            'blend_demand':{
                'j1': {},
                'j2': {},
                'j3': {},
                'j4': {},
                'j5': {'p1':0, 'p2':0},
                'j6': {'p1':0, 'p2':0},
                'j7': {'p1':0, 'p2':0},
                'j8': {'p1':0, 'p2':0}
            },
            
            "tau": {"s1": 0, "s2": 0},
            
            "delta": {"p1": 0, "p2": 0}
        }
        """
        
        self.t += 1
        if self.t == self.T:
            self.done = True
        
        # Updating demands before sanitizing action (important to avoid off-by-one)
        
        action = action.tolist()
        action = reconstruct_dict(action, self.mapping_act) # From non-human-readable list to human-readable dict
        action = self.sanitize_action_structure(action) # Modify action according to chosen rules (ex: set flows < 0.1 to 0 etc)
        
        
        prev_blend_invs = self.state["blenders"]
        
        for p in self.demands:
            self.state["demands_avail_next"][p] = self.delta0[p][self.t]
        
        for s in self.sources:
            self.state["sources_avail_next"][s] = self.tau0[s][self.t]
        
        # for s in self.sources:
        #     self.state["sources"][s] = self.state["sources"][s] \
        #                                 + min(action["tau"][s], self.state["sources_avail_next"][s]) \
        #                                 - sum([action["source_blend"][s][j] for j in action["source_blend"][s].keys()])

        # print("xxx",self.state,"\n\n", action)
        for s in self.sources:
            self.state["sources"][s] = self.state["sources"][s] \
                                        + action["tau"][s] \
                                        - sum([action["source_blend"][s][j] for j in action["source_blend"][s].keys()])
        
        for j in self.blenders:
            in_flow_sources = in_flow_blend = out_flow_blend = out_flow_demands = 0
            for s in self.sources:
                if j in action["source_blend"][s].keys():
                    in_flow_sources += action["source_blend"][s][j]
            for jp in self.blenders:
                # print(action["blend_blend"].keys(), action["blend_blend"][jp].keys())
                if j in action["blend_blend"][jp].keys():
                    in_flow_blend += action["blend_blend"][jp][j]
                if jp in action["blend_blend"][j].keys():
                    out_flow_blend += action["blend_blend"][j][jp]
            for p in self.demands:
                if p in action["blend_demand"][j].keys():
                    out_flow_demands += action["blend_demand"][j][p]
                
            self.state["blenders"][j] = self.state["blenders"][j] + in_flow_sources + in_flow_blend - out_flow_blend - out_flow_demands
                                            
        for p in self.demands:
            in_flow_blend = 0
            for jp in self.blenders:
                if p in action["blend_demand"][jp].keys():
                    in_flow_blend += action["blend_demand"][jp][p]
            
            self.state["demands"][p] = self.state["demands"][p] - action["delta"][p] + in_flow_blend
                                        
        for j in self.blenders:
            for q in self.properties:
                if self.state["blenders"][j] == 0:
                    self.state["properties"][j][q] = 0
                else:
                    in_flow_sources = in_flow_blend = out_flow_blend = out_flow_demands = 0
                    for s in self.sources:
                        if j in action["source_blend"][s].keys():
                            in_flow_sources += action["source_blend"][s][j] * self.sigma[s][q]
                    for jp in self.blenders:
                        if j in action["blend_blend"][jp].keys():
                            in_flow_blend += action["blend_blend"][jp][j] * self.state["properties"][jp][q]
                        if jp in action["blend_blend"][j].keys():
                            out_flow_blend += action["blend_blend"][j][jp] * self.state["properties"][jp][q]
                    for p in self.demands:
                        if p in action["blend_demand"][j].keys():
                            out_flow_demands += action["blend_demand"][j][p] * self.state["properties"][j][q]

                    self.state["properties"][j][q] = (1/self.state["blenders"][j]) * ( \
                                                    self.state["properties"][j][q] * prev_blend_invs[j] \
                                                    + in_flow_sources + in_flow_blend - out_flow_blend - out_flow_demands
                                                )
        
        self.update_reward(action)
        
        
        self.flatt_state, _ = flatten_and_track_mappings(self.state)
        
        for k in range(self.flatt_state.shape[0]):
            self.flatt_state[k] = max(0, self.flatt_state[k]) # non-negative state
        # Other sanitization to come, if we choose to implement max inventory 
        
        return self.flatt_state, self.reward, self.done, False, {"dict_state": self.state}
    
    def get_new_start_state_deterministic(self):
        self.tau0   = {'s1': [10, 10, 10, 0, 0, 0, 0],  's2': [30, 30, 30, 0, 0, 0, 0]} # An extra 0 to avoid indexerror (because the first value has to be in the starting state)
        self.delta0 = {'p1': [0, 0, 15, 15, 15, 15, 0], 'p2': [0, 0, 15, 15, 15, 15, 0]}
        
        self.state = {
            "sources": {s:0 for s in self.sources},
            "blenders": {b:0 for b in self.blenders},
            "demands": {p:0 for p in self.demands},
            "properties": {b: {q:0 for q in self.properties} for b in self.blenders},
            # How much we can buy at the current timestamp. Noted as "F^{IN}_{s,t}" in the paper
            "sources_avail_next": {s: self.tau0[s][0]   for s in self.sources},
            # How much we can sell at the current timestamp. Noted as "FD^L_{p,t}" in the paper
            "demands_avail_next": {p: self.delta0[p][0] for p in self.demands}
        }
    
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
            "sources_avail_next": {s: self.tau0[s][0]   for s in self.sources},
            # How much we can sell at the current timestamp. Noted as "FD^L_{p,t}" in the paper
            "demands_avail_next": {p: self.delta0[p][0] for p in self.demands}  
        }
        
    
    def update_reward(self, action):
        """
            Follows the definition/structure of the Overleaf Document

        Args:
            action (dict): See action_sample.json .
        """
        Q_float = Q_bin = 0
        
        for k in ["source_blend", "blend_blend", "blend_demand"]:
            for tank1 in action[k].keys():
                for tank2 in action[k][tank1].keys():
                    Q_float += action[k][tank1][tank2]
                    Q_bin += 1 if action[k][tank1][tank2] > 0 else 0 
                    
        Q = self.alpha * Q_bin + self.beta * Q_float
        
        R1 = -Q
        for p in self.demands:
            R1 += self.betaT_d[p] * action["delta"][p] * self.Z
        for s in self.sources:
            R1 -= self.betaT_s[s] * action["tau"][s] * 1        # 1 for now: no need to emphasize buying
        
        R2 = R1
        for j in self.blenders:
            R2 -= self.penalty_in_out_flow(j, action)
            for q in self.properties:
                for p in self.demands:
                    R2 -= self.penalty_quality(p, q, j, action)
                
        self.reward += R2
        
    def penalty_quality(self, p, q, j, action):
        if self.state["properties"][j][q] < self.sigma_lb[p][q] and (p in action["blend_demand"][j].keys() and action["blend_demand"][j][p]) > 0:
            self.logg(f"Sold qualities not in required bounds ! ; ctxt: {p} {q} {j}")
            return self.M
        return 0
    
    def penalty_in_out_flow(self, j, action):
        sum_in = sum_out = 0
        for jp in self.blenders:
            sum_in  += action["blend_blend"][jp][j] if j in action["blend_blend"][jp].keys() else 0
            sum_out += action["blend_blend"][j][jp] if jp in action["blend_blend"][j].keys() else 0
        
        for s in self.sources:
            sum_in  += action["source_blend"][s][j] if j in action["source_blend"][s].keys() else 0
        
        for p in self.demands:
            sum_out += action["blend_demand"][j][p] if p in action["blend_demand"][j].keys() else 0
            
        if sum_in > 0 and sum_out > 0: # /!\
            self.logg(f"In and out flow both non-zero ! tank: {j} ; t: {self.t} ; ")
            return self.M
        
        return 0
    
    # Penalties to add (?):
    #   - Flow > Inventory (or let the model learn by itself and we cap the action before applying it ?)
    #   - Sell > demand
    #   - Buy > Supply
    
    def sanitize_action_structure(self, action):
        """Normalize model action if needed
        
        Incur penalties if the action leads to impossible/unauthorized states

        Args:
            action (dict): Action dict
        """
        self.logg("sanitizing action structure...", action)
        # Adding all blender/blender pairs to avoid keyerror in self.step()
        
        # for s in self.sources:
        #     for j in self.blenders:
        #         if j not in action["source_blend"][s].keys():
        #             action["source_blend"][s][j] = 0
        
        
        for j in self.blenders:
            if j not in action["blend_blend"].keys():
                action["blend_blend"][j] = {}
            # else:
            #     for jp in self.blenders:
            #         if jp not in action["blend_blend"][j].keys():
            #             action["blend_blend"][j][jp] = 0

            if j not in action["blend_demand"].keys():
                action["blend_demand"][j] = {}
        #     else:
        #         for p in self.demands:
        #             if p not in action["blend_demand"][j].keys():
        #                 action["blend_demand"][j][p] = 0
        return(action)
    
    
    def sanitize_action_flows(self, action, pen = True):
        ### Penalty section ###
        
        # if log: f = open("logg_penalties", "+a")
        
        # Add penalty and log if trying to buy too much product
        for s in self.sources:
            if action["tau"][s] > self.state["sources_avail_next"][s]:
                action["tau"][s] = self.state["sources_avail_next"][s]
                self.reward -= self.P if pen else 0 # incur penalty
                self.logg(f"{self.t}; {s}: bought too much")
        
        # Add penalty and log if trying to sell too much product (more than available demand or more than available inventory)
        for p in self.demands:
            if action["delta"][p] > self.state["demands_avail_next"][p]:
                action["delta"][p] = self.state["demands_avail_next"][p]
                self.reward -= self.P if pen else 0 # incur penalty
                self.logg(f"{self.t}; {p}: sold too much (more than demand)")
                
            if action["delta"][p] > self.state["demands"][p]:
                action["delta"][p] = self.state["demands"][p]
                self.reward -= self.P if pen else 0 # incur penalty
                self.logg(f"{self.t}; {p}: sold too much (more than inv)")
        
        # Add penalty and log if trying to outflow too much matter from a tank
        for j in self.blenders:
            S = sum(action["blend_blend"][j].values()) + sum(action["blend_demand"][j].values())
            if S > self.state["blenders"][j]:
                self.reward -= self.P if pen else 0 # incur penalty
                self.logg(f"{self.t} ; {j}: outgoing flow higher than inventory")
                
        # self.logg("\n\n")
        # if log: f.close()
        
        # print("sanitized action!!")
        
        return action
    
    
    def load_gurobi_actions(self): # No "tau" !
        s_b_flows = np.load("arrays/s_b_flows.npy")
        b_b_flows = np.load("arrays/b_b_flows.npy")
        b_d_flows = np.load("arrays/b_d_flows.npy")
        delta = np.load("arrays/delta.npy")
        
        return s_b_flows, b_b_flows, b_d_flows, delta
        
    def reset(self, seed=0):
        self.t = 0
        
        if self.determ:
            self.get_new_start_state_deterministic()
        else:
            self.get_new_start_state_probabilistic(seed)
            
        self.reward = 0
        self.done = False
        
        self.flatt_state, _ = flatten_and_track_mappings(self.state)
        return self.flatt_state, {"dict_state": self.state}
        
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