import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from or_gym.utils import assign_env_config


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