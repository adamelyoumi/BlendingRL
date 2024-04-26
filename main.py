import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from or_gym.utils import assign_env_config
import json


class Tank():
    def __init__(self) -> None:
        pass


class SupplyTank(Tank):
    def __init__(self) -> None:
        super().__init__()

class PoolTank(Tank):
    def __init__(self) -> None:
        super().__init__()
    
class DemandTank(Tank):
    def __init__(self) -> None:
        super().__init__()



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
            self.observation_space = spaces.Dict({
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
        
        self.M = 100
        self.T = 7
        self.alpha = 0.1
        self.beta = 0.02
        
        with open("./connections_sample.json" ,"r") as f:
            connections_s = f.readline()
        self.connections = json.loads(connections_s)
        
        self.properties = ["q1"]
        
        self.tau0   = {'s1': [10, 10, 10, 0, 0, 0, 0], 's2': [30, 30, 30, 0, 0, 0, 0]}
        self.delta0 = {'p1': [0, 0, 0, 15, 15, 15, 15], 'p2': [0, 0, 0, 15, 15, 15, 15]}
        
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
        
        
        self.sources = list(self.connections["tau"].keys())
        self.demands = list(self.connections["blend_blend"].keys())
        self.blenders = list(self.connections["delta"].keys())
        
        assign_env_config(self, kwargs)
        
        
        self.start_state = {
            "sources": {s:0 for s in self.sources},
            "blenders": {b:0 for b in self.blenders},
            "demands": {p:0 for p in self.demands},
            "properties": {b: {q:0 for q in self.properties} for b in self.blenders},
            # How much we can buy at the current timestamp. Noted as "F^{IN}_{s,t}" in the paper
            "sources_avail": {s:{q:0 for q in self.properties} for s in self.sources},
            # How much we can sell at the current timestamp. Noted as "FD^L_{p,t}" in the paper
            "demands_avail": {p:{q:0 for q in self.properties} for p in self.demands}  
        }
        
        self.reset()
    
    def step(self, action):
        if self.t == self.T:
            self.done = True
            return self.state, self.reward, self.done
        
        self.t += 1
        
        prev_blend_invs = self.state["blenders"]
        
        for s in self.sources:
            self.state["sources"][s] = self.state["sources"][s] \
                                        + min(action["tau"][s], self.state["sources_avail"]) \
                                        - sum([action["source_blend"][s][j] for j in action[s].keys()])
        
        for j in self.blenders:
            self.state["blenders"][j] = self.state["blenders"][j] \
                                        + sum([action[s][j] for s in action["source_blend"].keys()]) \
                                        + sum([action[jp][j] for jp in action["blend_blend"].keys()]) \
                                        - sum([action[j][jp] for jp in action["blend_blend"][j].keys()]) \
                                        - sum([action[j][p] for p in action["blend_demand"][j].keys()]) \
                                            
        for p in self.demands:
            self.state["demands"][p] = self.state["demands"][p] \
                                        - action["delta"][p] \
                                        + sum([action[j][p] for j in action["blend_demand"].keys()])
                                        
        for j in self.blenders:
            for q in self.properties:
                self.state["properties"][j][q] = (1/self.state["blenders"][j]) * ( \
                                                    self.state["properties"][j][q] * prev_blend_invs[j] \
                                                    + sum(self.sigma[s][q] * action[s][j] for s in self.sources) \
                                                    + sum(self.state["properties"][jp][q] * action[jp][j] for jp in self.blenders) \
                                                    - sum(self.state["properties"][j][q] * action[j][jp] for jp in self.blenders) \
                                                    - sum(self.state["properties"][j][q] * action[j][p] for p in self.demands)
                                                )
        
        # Telling the model how much can be bought/sold
        for q in self.properties:
            for s in self.sources:
                self.state["sources_avail"][s][q] = self.tau0[self.t]
            
            for p in self.demands:
                self.state["demands_avail"][p][q] = self.delta0[self.t]
                
        
        self.update_reward(action)
        
        return self.state, self.reward, self.done
        
    def update_reward(self, action):
        """
            Follows the definition/structure of the Overleaf Document

        Args:
            action (dict): See action_sample.json .
        """
        Q_float = Q_bin = R1 = R2 = 0
        
        for k in ["source_blend", "blend_blend", "blend_demand"]:
            for tank1 in action[k].keys():
                for tank2 in action[k][tank1].keys():
                    Q_float += action[k][tank1][tank2]
                    Q_bin += 1 if action[k][tank1][tank2] > 0 else 0 
                    
        Q = self.alpha * Q_bin + self.beta * Q_float
        
        R1 = 0
        for p in self.demands:
            R1 += self.betaT_d[p] * min(action["delta"][p], self.state["demands"][p])
        for s in self.sources:
            R1 -= self.betaT_s[s] * action["tau"][s]
            
        R1 = R1 - Q
        
        for j in self.blenders:
            R2 += self.state(j, action)
            for q in self.properties:
                for p in self.demands:
                    R2 += self._penalty_quality(p, q, j, action)
        
        R2 = R2 + R1
        
        self.reward = R2
        
    def _penalty_quality(self, p, q, j, action):
        if self.state["properties"][j][q] < self.sigma_lb[p][q] and action["blend_demand"][j][p] > 0:
            return self.M
        return 0
    
    def _penalty_in_out_flow(self, j, action):
        sum_in = sum_out = 0
        for jp in self.blenders:
            sum_in  += action["blend_blend"][jp][j]
            sum_out += action["blend_blend"][j][jp]
        
        for s in self.sources:
            sum_in  += action["source_blend"][s][j]
        
        for p in self.demands:
            sum_out += action["blend_demand"][j][p]
            
        if sum_in > 0 and sum_out > 0:
            return self.M
        
        return 0
        
    def reset(self):
        self.t = 0
        self.state = self.start_state
        self.reward = 0
        self.done = False
        
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