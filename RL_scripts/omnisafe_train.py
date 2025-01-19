
import sys, os

curr_dir = os.path.abspath(os.getcwd())
sys.path.append(curr_dir)

import omnisafe
from utils import *

layout = "simple"
ALGO = "CPO"
cfg = cfg_to_omni(73, layout, total_steps=1e6, algo=ALGO, device="cpu")

env_id = f'Blend-{layout}'
agent = omnisafe.Agent(ALGO, env_id, custom_cfgs=cfg)
agent.learn()
