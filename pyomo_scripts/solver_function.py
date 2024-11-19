
import sys, os

curr_dir = os.path.abspath(os.getcwd())
sys.path.append(curr_dir)

import numpy as np
import pandas as pd
import random as rd
import datetime, time
import math as m
import argparse

from pyomo.environ import *
from utils import *


tau0 = {"s1": [10]*6 + [0], "s2": [10]*6 + [0]}
delta0 = {"p1": [0] + [10]*6, "p2": [0] + [10]*6}


T = 6
alpha = 0.1
beta = 0.1
sigma = {"s1":{"q1": 0.06}, "s2":{"q1": 0.26}}
sigma_ub = {"p1":{"q1": 0.16}, "p2":{"q1": 1}}
sigma_lb = {"p1":{"q1": 0}, "p2":{"q1": 0}}
s_inv_ub = {'s1': 30, 's2': 30}
d_quals_lb = {'p1': 0, 'p2': 0}
d_quals_ub = {'p1': 0.16, 'p2': 0.1}
d_inv_ub = {'p1': 30, 'p2': 30}
betaT_d = {'p1': 2, 'p2': 1}

properties = ["q1"]
timestamps = list(range(T))
b_inv_ub = {"j1": 30, "j2": 30, "j3": 30, "j4": 30}

def solve(tau0, delta0, layout,
        alpha = alpha,
        beta = beta,
        properties = properties,
        timestamps = timestamps,
        sigma = sigma,
        sigma_ub = sigma_ub,
        sigma_lb = sigma_lb,
        s_inv_ub = s_inv_ub,
        d_quals_lb = d_quals_lb,
        d_quals_ub = d_quals_ub,
        d_inv_ub = d_inv_ub,
        betaT_d = betaT_d,
        b_inv_ub = b_inv_ub,
        mingap = 0.005, maxtime = 30):
    
    connections, action_sample = get_jsons(layout)
    sources, blenders, demands = get_sbp(connections)
    
    # Model
    model = ConcreteModel()

    # Sets
    model.sources = Set(initialize=sources)
    model.demands = Set(initialize=demands)
    model.blenders = Set(initialize=blenders)
    model.properties = Set(initialize=properties)
    model.timestamps = Set(initialize=timestamps)

    # Parameters
    model.s_inv_ub = Param(model.sources, initialize=s_inv_ub)
    model.tau0 = Param(model.sources, initialize=tau0)
    model.d_quals_lb = Param(model.demands, initialize=d_quals_lb)
    model.d_quals_ub = Param(model.demands, initialize=d_quals_ub)
    model.d_inv_ub = Param(model.demands, initialize=d_inv_ub)
    model.delta0 = Param(model.demands, initialize=delta0)
    model.betaT_d = Param(model.demands, initialize=betaT_d)
    model.b_inv_ub = Param(model.blenders, initialize=b_inv_ub)

    # Decision variables
    model.source_inv_before_flow = Var(model.sources, model.timestamps, domain=NonNegativeReals) # Before flow but after buy
    model.source_inv = Var(model.sources, model.timestamps, domain=NonNegativeReals)
    model.blend_inv = Var(model.blenders, model.timestamps, domain=NonNegativeReals)
    model.demand_inv = Var(model.demands, model.timestamps, domain=NonNegativeReals)
    model.demand_inv_after_sell = Var(model.demands, model.timestamps, domain=NonNegativeReals)

    model.demand_sold = Var(model.demands, model.timestamps, domain=NonNegativeReals) # Represents the amount of product sold at each timestep; necessary for objective function
    model.offer_bought = Var(model.sources, model.timestamps, domain=NonNegativeReals) # Represents the amount of product sold at each timestep; necessary for objective function

    model.prop_blend_inv = Var(model.properties, model.blenders, model.timestamps, domain=NonNegativeReals)

    model.source_blend_flow = Var(model.sources, model.blenders, model.timestamps, domain=NonNegativeReals)
    model.blend_blend_flow = Var(model.blenders, model.blenders, model.timestamps, domain=NonNegativeReals)
    model.blend_demand_flow = Var(model.blenders, model.demands, model.timestamps, domain=NonNegativeReals)

    model.source_blend_bin = Var(model.sources, model.blenders, model.timestamps, domain=Binary)
    model.blend_blend_bin = Var(model.blenders, model.blenders, model.timestamps, domain=Binary)
    model.blend_demand_bin = Var(model.blenders, model.demands, model.timestamps, domain=Binary)

    # flow = 0 if the pair is not in the dict connections
    def connections_rule0_1(model, s, j, t):
        if j not in connections["source_blend"][s]:
            return model.source_blend_flow[s, j, t] == 0
        else:
            return model.source_blend_flow[s, j, t] >= 0
        
    def connections_rule0_3(model, j, p, t):
        if p not in connections["blend_demand"][j]:
            return model.blend_demand_flow[j, p, t] == 0
        else:
            return model.blend_demand_flow[j, p, t] >= 0
        
    def connections_rule0_4(model, j1, j2, t):
        if j2 not in connections["blend_blend"][j1]:
            return model.blend_blend_flow[j1, j2, t] == 0
        else:
            return model.blend_blend_flow[j1, j2, t] >= 0
    
    model.material_balance_rule0_1 = Constraint(model.sources,  model.blenders, model.timestamps, rule=connections_rule0_1)
    # model.material_balance_rule0_2 = Constraint(model.sources,  model.demands,  model.timestamps, rule=connections_rule0_2)
    model.material_balance_rule0_3 = Constraint(model.blenders, model.demands,  model.timestamps, rule=connections_rule0_3)
    model.material_balance_rule0_4 = Constraint(model.blenders, model.blenders, model.timestamps, rule=connections_rule0_4)

    # Inventory bounds
    def connections_rule0_1_1(model, j, t):
        return model.blend_inv[j, t] <= model.b_inv_ub[j]

    model.material_balance_rule0_1_1 = Constraint(model.blenders, model.timestamps, rule=connections_rule0_1_1)

    # Cannot buy more than what is available
    def material_balance_rule1_0(model, s, t):
        return model.offer_bought[s, t] <= model.tau0[s][t]

    # Cannot buy more than available space in tank
    def material_balance_rule1_1(model, s, t):
        if t == 0:
            return model.offer_bought[s, t] <= model.s_inv_ub[s] # To avoid looking up source inv at t = -1
        else:
            return model.offer_bought[s, t] <= model.s_inv_ub[s] - model.source_inv[s, model.timestamps.prev(t)]

    # Updating source inv before outgoing flows but after buy
    def material_balance_rule1_2(model, s, t):
        if t == 0:
            return model.source_inv_before_flow[s, t] == model.offer_bought[s, t] # Initialize inventory at t=0
        else:
            return model.source_inv_before_flow[s, t] == model.source_inv[s, model.timestamps.prev(t)] \
                                                        + model.offer_bought[s, t]

    # Updating source after outgoing flows and after buy inv
    def material_balance_rule1_3(model, s, t):
        return model.source_inv[s, t] == model.source_inv_before_flow[s, t] \
                                        - sum(model.source_blend_flow[s, j, t] for j in model.blenders)

    model.material_balance_rule1_0 = Constraint(model.sources, model.timestamps, rule=material_balance_rule1_0)
    model.material_balance_rule1_1 = Constraint(model.sources, model.timestamps, rule=material_balance_rule1_1)
    model.material_balance_rule1_2 = Constraint(model.sources, model.timestamps, rule=material_balance_rule1_2)
    model.material_balance_rule1_3 = Constraint(model.sources, model.timestamps, rule=material_balance_rule1_3)

    # Updating blender inventories
    def material_balance_rule2(model, j, t):
        if t == 0:  # Initialize inventory at t=0
            return model.blend_inv[j, t] == sum(model.source_blend_flow[s, j, t] for s in model.sources)
        else:
            return model.blend_inv[j, t] == model.blend_inv[j, model.timestamps.prev(t)] \
                                        + sum(model.source_blend_flow[s, j, t] for s  in model.sources) \
                                        + sum(model.blend_blend_flow[jp, j, t] for jp in model.blenders) \
                                        - sum(model.blend_blend_flow[j, jp, t] for jp in model.blenders) \
                                        - sum(model.blend_demand_flow[j, d, t] for d  in model.demands)

    model.material_balance_rule2 = Constraint(model.blenders, model.timestamps, rule=material_balance_rule2)

    # Cannot sell more than what is asked
    def material_balance_rule3_0(model, p, t):
        return model.demand_sold[p, t] <= model.delta0[p][t]

    # Cannot sell more than what is available
    def material_balance_rule3_1(model, p, t):
        return model.demand_sold[p, t] <= model.demand_inv[p, t]

    # Updating demand before sell inv
    def material_balance_rule3_2(model, p, t):
        if t == 0:
            return model.demand_inv[p, t] == 0 # Initialize inventory at t=0
        else:
            return model.demand_inv[p, t] == model.demand_inv_after_sell[p, model.timestamps.prev(t)] \
                                        + sum(model.blend_demand_flow[j, p, t] for j in model.blenders)

    # Updating demand after sell inv
    def material_balance_rule3_3(model, p, t):
        if t == 0:
            return model.demand_inv_after_sell[p, t] == 0 # Initialize inventory at t=0
        else:
            return model.demand_inv_after_sell[p, t] == model.demand_inv[p, t] - model.demand_sold[p, t] 

    model.material_balance_rule3_0 = Constraint(model.demands, model.timestamps, rule=material_balance_rule3_0)
    model.material_balance_rule3_1 = Constraint(model.demands, model.timestamps, rule=material_balance_rule3_1)
    model.material_balance_rule3_2 = Constraint(model.demands, model.timestamps, rule=material_balance_rule3_2)
    model.material_balance_rule3_3 = Constraint(model.demands, model.timestamps, rule=material_balance_rule3_3)

    M = 90
    # in/out flow constraints
    def material_balance_rule4_1(model, s, j, t):
        return model.source_blend_flow[s, j, t] <= M * model.source_blend_bin[s, j, t]

    def material_balance_rule4_2(model, j1, j2, t):
        return model.blend_blend_flow[j1, j2, t] <= M * model.blend_blend_bin[j1, j2, t]

    def material_balance_rule4_3(model, j, p, t):
        return model.blend_demand_flow[j, p, t] <= M * model.blend_demand_bin[j, p, t]

    model.material_balance_rule4_1 = Constraint(model.sources, model.blenders,  model.timestamps, rule=material_balance_rule4_1)
    model.material_balance_rule4_2 = Constraint(model.blenders, model.blenders, model.timestamps, rule=material_balance_rule4_2)
    model.material_balance_rule4_3 = Constraint(model.blenders, model.demands,  model.timestamps, rule=material_balance_rule4_3)

    # in/out flow constraints
    def material_balance_rule5_1(model, s, j, p, t):
        return model.source_blend_bin[s, j, t] <= 1 - model.blend_demand_bin[j, p, t]

    def material_balance_rule5_2(model, s, j1, j2, t):
        return model.source_blend_bin[s, j1, t] <= 1 - model.blend_blend_bin[j1, j2, t]

    def material_balance_rule5_3(model, j1, j2, p, t):
        return model.blend_blend_bin[j1, j2, t] <= 1 - model.blend_demand_bin[j2, p, t]

    model.material_balance_rule5_1 = Constraint(model.sources, model.blenders, model.demands,  model.timestamps, rule=material_balance_rule5_1)
    model.material_balance_rule5_2 = Constraint(model.sources, model.blenders, model.blenders, model.timestamps, rule=material_balance_rule5_2)
    model.material_balance_rule5_3 = Constraint(model.blenders, model.blenders, model.demands, model.timestamps, rule=material_balance_rule5_3)

    # Quality calculations
    def material_balance_rule6(model, q, j, t):
        if t == 0:
            return model.prop_blend_inv[q, j, t] * model.blend_inv[j, t] == sum(sigma[s][q] * model.source_blend_flow[s, j, t] for s in model.sources) # Initialize empty inventory at t=0
        else:
            return model.prop_blend_inv[q, j, t] * model.blend_inv[j, t] == model.prop_blend_inv[q, j, model.timestamps.prev(t)] * model.blend_inv[j, model.timestamps.prev(t)] \
                                                                            + sum(sigma[s][q] * model.source_blend_flow[s, j, t] for s in model.sources) \
                                                                            + sum(model.prop_blend_inv[q, jp, t] * model.blend_blend_flow[jp, j, t] for jp in model.blenders) \
                                                                            - sum(model.prop_blend_inv[q, j,  t] * model.blend_blend_flow[j, jp, t] for jp in model.blenders) \
                                                                            - sum(model.prop_blend_inv[q, j,  t] * model.blend_demand_flow[j, p, t] for p in model.demands)

    model.material_balance_rule6 = Constraint(model.properties, model.blenders, model.timestamps, rule=material_balance_rule6)

    # Quality constraints
    def material_balance_rule7_1(model, q, p, j, t):
        return sigma_lb[p][q] - M * (1 - model.blend_demand_bin[j, p, t]) <= model.prop_blend_inv[q, j, t]

    def material_balance_rule7_2(model, q, p, j, t):
        return sigma_ub[p][q] + M * (1 - model.blend_demand_bin[j, p, t]) >= model.prop_blend_inv[q, j, t]

    model.material_balance_rule7_1 = Constraint(model.properties, model.demands, model.blenders, model.timestamps, rule=material_balance_rule7_1)
    model.material_balance_rule7_2 = Constraint(model.properties, model.demands, model.blenders, model.timestamps, rule=material_balance_rule7_2)

    def obj_function(model):
        return  sum( sum(model.betaT_d[p] * (model.demand_sold[p, t]) for p in model.demands) for t in range(T)) \
                - sum( sum(
                        sum(alpha * model.source_blend_bin[s, j, t] + beta * model.source_blend_flow[s, j, t] for s in model.sources) \
                        + sum(alpha * model.blend_blend_bin[j, jp, t] + beta * model.blend_blend_flow[j, jp, t] for jp in model.blenders) \
                        + sum(alpha * model.blend_demand_bin[j, p, t] + beta * model.blend_demand_flow[j, p, t] for p in model.demands) 
                    for t in model.timestamps) 
                for j in model.blenders)

    model.obj = Objective(rule=obj_function, sense=maximize)

    # Solve the model
    solver = SolverFactory('gurobi')
    solver.options['timelimit'] = maxtime # Stop if runtime > 30s
    solver.options['mipgap'] = mingap # Stop if gap < 0.5%

    result = solver.solve(model, tee=False)
    
    solveinfo = result.json_repn()
    
    return model, result, solveinfo

if __name__ == "__main__":
    model, r, info = solve(tau0, delta0, "simple")
    print(r)

