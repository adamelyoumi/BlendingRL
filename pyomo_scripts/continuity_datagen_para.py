import sys, os

if os.name == "nt":
    os.chdir("C:\\Users\\adame\\OneDrive\\Bureau\\CODE\\BlendingRL")
else:
    os.chdir("/home/ubuntu/bp")

sys.path.append(os.getcwd())

import numpy as np
from pyomo.environ import *
from utils import *
from pyomo_scripts.solver_function import solve
import pickle
import time
from multiprocessing import Pool, cpu_count


def process_point(args):
    p1, p2 = args
    
    sigma = {"s1":{"q1": 0.06}, "s2":{"q1": 0.26}}
    sigma_ub = {"p1":{"q1": 0.16}, "p2":{"q1": 1}}
    sigma_lb = {"p1":{"q1": 0}, "p2":{"q1": 0}}
    betaT_d = {'p1': 2, 'p2': 1}
    betaT_s = {'s1': 0.2, 's2': 0.2}
    alpha = 0.1
    beta = 0.01
    
    tau0 = {"s1": [10]*5 + [0], "s2": [10]*5 + [0]}
    delta0 = {"p1": [0, p1] + [10]*4, "p2": [0, p2] + [10]*4}
    
    mingap = 0
    maxtime = 120
    coef_sym = 0.001
    
    model, r, info = solve(tau0, delta0, "simple",
                            sigma = sigma, sigma_ub = sigma_ub, sigma_lb = sigma_lb,
                            betaT_d = betaT_d, betaT_s = betaT_s,
                            alpha = alpha, beta = beta,        
                            mingap = mingap, maxtime = maxtime)
    
    gap = (info["Problem"][0]["Upper bound"]/info["Problem"][0]["Lower bound"])-1 if info["Problem"][0]["Lower bound"] != 0 else 1
    gap = gap.__round__(3)
    print(info["Solver"][0]["Status"], gap, tau0, delta0, "\nbought:", 
            [model.offer_bought["s1", t].value for t in range(6)], [model.offer_bought["s2", t].value for t in range(6)], 
            "\nsold:", 
            [model.demand_sold["p1", t].value for t in range(6)], [model.demand_sold["p2", t].value for t in range(6)], '\n')
    
    key = f"{p1};{p2}"
    
    result = {}
    if info["Solver"][0]["Status"] == "ok":
        result["res_s1"] = [model.offer_bought["s1", t].value for t in range(6)]
        result["res_s2"] = [model.offer_bought["s2", t].value for t in range(6)]
        result["res_flow_s1_j1"] = [model.source_blend_flow["s1", "j1", t].value for t in range(6)]
        result["res_flow_s1_j2"] = [model.source_blend_flow["s1", "j2", t].value for t in range(6)]
        result["res_flow_s1_j3"] = [model.source_blend_flow["s1", "j3", t].value for t in range(6)]
        result["res_flow_s1_j4"] = [model.source_blend_flow["s1", "j4", t].value for t in range(6)]
        result["res_flow_s2_j1"] = [model.source_blend_flow["s2", "j1", t].value for t in range(6)]
        result["res_flow_s2_j2"] = [model.source_blend_flow["s2", "j2", t].value for t in range(6)]
        result["res_flow_s2_j3"] = [model.source_blend_flow["s2", "j3", t].value for t in range(6)]
        result["res_flow_s2_j4"] = [model.source_blend_flow["s2", "j4", t].value for t in range(6)]
        result["res_flow_j1_p1"] = [model.blend_demand_flow["j1", "p1", t].value for t in range(6)]
        result["res_flow_j2_p1"] = [model.blend_demand_flow["j2", "p1", t].value for t in range(6)]
        result["res_flow_j3_p1"] = [model.blend_demand_flow["j3", "p1", t].value for t in range(6)]
        result["res_flow_j4_p1"] = [model.blend_demand_flow["j4", "p1", t].value for t in range(6)]
        result["res_flow_j1_p2"] = [model.blend_demand_flow["j1", "p2", t].value for t in range(6)]
        result["res_flow_j2_p2"] = [model.blend_demand_flow["j2", "p2", t].value for t in range(6)]
        result["res_flow_j3_p2"] = [model.blend_demand_flow["j3", "p2", t].value for t in range(6)]
        result["res_flow_j4_p2"] = [model.blend_demand_flow["j4", "p2", t].value for t in range(6)]
        result["res_p1"] = [model.demand_sold["p1", t].value for t in range(6)]
        result["res_p2"] = [model.demand_sold["p2", t].value for t in range(6)]
        
    elif info["Solver"][0]["Status"] == "aborted":
        for k in ["res_s1","res_s2",
                  "res_flow_s1_j1","res_flow_s1_j2","res_flow_s1_j3","res_flow_s1_j4",
                  "res_flow_s2_j1","res_flow_s2_j2","res_flow_s2_j3","res_flow_s2_j4",
                  "res_flow_j1_p1","res_flow_j2_p1","res_flow_j3_p1","res_flow_j4_p1",
                  "res_flow_j1_p2","res_flow_j2_p2","res_flow_j3_p2","res_flow_j4_p2",
                  "res_p1","res_p2"]:
            result[k] = [None for _ in range(6)]
    
    return key, result

if __name__ == '__main__':
    len_x, len_y = 41,41
    x = np.linspace(9, 10, len_x)
    y = np.linspace(10, 11, len_y)
    X, Y = np.meshgrid(x, y)

    D = {k: {} for k in ["res_s1", "res_s2", 
                         "res_flow_s1_j1", "res_flow_s1_j2", "res_flow_s1_j3", "res_flow_s1_j4", 
                         "res_flow_s2_j1", "res_flow_s2_j2", "res_flow_s2_j3", "res_flow_s2_j4", 
                         "res_flow_j1_p1", "res_flow_j2_p1", "res_flow_j3_p1", "res_flow_j4_p1", 
                         "res_flow_j1_p2", "res_flow_j2_p2", "res_flow_j3_p2", "res_flow_j4_p2", 
                         "res_p1", "res_p2"]}

    points = [(p1, p2) for p1 in x for p2 in y]

    start_time = time.time()

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_point, points)

    for key, result in results:
        for k in D.keys():
            D[k][key] = result[k]

    print("Total time elapsed:", time.time() - start_time)

    with open(f'./cont_data_{len_x}x{len_y}.pkl', 'wb') as handle:
        pickle.dump(D, handle)