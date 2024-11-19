
import sys, os
curr_dir = os.path.abspath(os.getcwd())
sys.path.append(curr_dir)
sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))
os.chdir(os.path.join(os.getcwd(), ".."))

import numpy as np
from pyomo.environ import *
from utils import *
from solver_function import solve
import pickle
import time


x = np.linspace(9, 10, 11)
y = np.linspace(10, 11, 11)
X, Y = np.meshgrid(x, y)
x

D = {k: {} for k in ["res_s1", "res_s2", 
                     "res_flow_s1_j1", "res_flow_s1_j2", "res_flow_s1_j3", "res_flow_s1_j4", 
                     "res_flow_s2_j1", "res_flow_s2_j2", "res_flow_s2_j3", "res_flow_s2_j4", 
                     "res_flow_j1_p1", "res_flow_j2_p1", "res_flow_j3_p1", "res_flow_j4_p1", 
                     "res_flow_j1_p2", "res_flow_j2_p2", "res_flow_j3_p2", "res_flow_j4_p2", 
                     "res_p1", "res_p2"]}

mingap = 0.005
time_ = time.time()

start_time = time.time()

for i, p1 in enumerate(x): # Between 9 and 10
    print("time elapsed after iterating over y once:", time.time() - time_)
    time_ = time.time()
    for j, p2 in enumerate(y): # Between 10 and 11
        tau0 = {"s1": [10]*5 + [0], "s2": [10]*5 + [0]}
        delta0 = {"p1": [0, p1] + [10]*4, "p2": [0, p2] + [10]*4}
        model, r, info = solve(tau0, delta0, layout="simple", mingap=mingap, maxtime = 120)
        
        gap = (info["Problem"][0]["Upper bound"]/info["Problem"][0]["Lower bound"])-1 if info["Problem"][0]["Lower bound"] != 0 else 1
        gap = gap.__round__(3)
        print(info["Solver"][0]["Status"], gap, tau0, delta0, "\nbought:", 
                [model.offer_bought["s1", t].value for t in range(6)], [model.offer_bought["s2", t].value for t in range(6)], "\nsold:", 
                [model.demand_sold["p1", t].value for t in range(6)], [model.demand_sold["p2", t].value for t in range(6)], '\n')
        
        key = f"{p1};{p2}"
        
        D["res_s1"][key] = [model.offer_bought["s1", t].value for t in range(6)]
        D["res_s2"][key] = [model.offer_bought["s2", t].value for t in range(6)]
        D["res_flow_s1_j1"][key] = [model.source_blend_flow["s1", "j1", t].value for t in range(6)]
        D["res_flow_s1_j2"][key] = [model.source_blend_flow["s1", "j2", t].value for t in range(6)]
        D["res_flow_s1_j3"][key] = [model.source_blend_flow["s1", "j3", t].value for t in range(6)]
        D["res_flow_s1_j4"][key] = [model.source_blend_flow["s1", "j4", t].value for t in range(6)]
        D["res_flow_s2_j1"][key] = [model.source_blend_flow["s2", "j1", t].value for t in range(6)]
        D["res_flow_s2_j2"][key] = [model.source_blend_flow["s2", "j2", t].value for t in range(6)]
        D["res_flow_s2_j3"][key] = [model.source_blend_flow["s2", "j3", t].value for t in range(6)]
        D["res_flow_s2_j4"][key] = [model.source_blend_flow["s2", "j4", t].value for t in range(6)]
        D["res_flow_j1_p1"][key] = [model.blend_demand_flow["j1", "p1", t].value for t in range(6)]
        D["res_flow_j2_p1"][key] = [model.blend_demand_flow["j2", "p1", t].value for t in range(6)]
        D["res_flow_j3_p1"][key] = [model.blend_demand_flow["j3", "p1", t].value for t in range(6)]
        D["res_flow_j4_p1"][key] = [model.blend_demand_flow["j4", "p1", t].value for t in range(6)]
        D["res_flow_j1_p2"][key] = [model.blend_demand_flow["j1", "p2", t].value for t in range(6)]
        D["res_flow_j2_p2"][key] = [model.blend_demand_flow["j2", "p2", t].value for t in range(6)]
        D["res_flow_j3_p2"][key] = [model.blend_demand_flow["j3", "p2", t].value for t in range(6)]
        D["res_flow_j4_p2"][key] = [model.blend_demand_flow["j4", "p2", t].value for t in range(6)]
        D["res_p1"][key] = [model.demand_sold["p1", t].value for t in range(6)]
        D["res_p2"][key] = [model.demand_sold["p2", t].value for t in range(6)]

with open('./cont_data.pickle', 'wb') as handle:
    pickle.dump(D, handle)
    

print("Total time elapsed:", time.time() - start_time)