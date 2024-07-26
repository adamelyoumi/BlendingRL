import numpy as np
import pandas as pd
from itertools import product
import random as rd
import torch as th
import datetime

from pyomo.environ import *

N = 50

T = 6
alpha = 0.1
beta = 1

sources = ["s1", "s2"]
demands = ["d1", "d2"]
blenders = ["b1", "b2", "b3", "b4"]
properties = ["p1"]
timestamps = list(range(T))


sigma = {"s1":{"p1": 0.06}, "s2":{"p1": 0.26}}

sigma_lb = {"d1":{"p1": 0}, "d2":{"p1": 0}}
sigma_ub = {"d1":{"p1": 0.16}, "d2":{"p1": 1}}

s_inv_lb = {'s1': 0, 's2': 0}
s_inv_ub = {'s1': 0, 's2': 0}

d_quals_lb = {'d1': 0, 'd2': 0}
d_quals_ub = {'d1': 0.16, 'd2': 0.1}

d_inv_lb = {'d1': 0, 'd2': 0}
d_inv_ub = {'d1': 0, 'd2': 0}
betaT_d = {'d1': 2, 'd2': 1}

connections = {
    'source_blend':{'s1': ['b1', 'b2', 'b3', 'b4'], 's2': ['b1', 'b2', 'b3', 'b4']},
    'source_demand':{'s1': [], 's2': []},
    'blend_blend':{'b1': [], 'b2': [], 'b3': [], 'b4': []},
    'blend_demand':{
        'b1': ['d1', 'd2'],
        'b2': ['d1', 'd2'],
        'b3': ['d1', 'd2'],
        'b4': ['d1', 'd2']
    }
}

cols_obs = [
    'sources_s1', 'sources_s2',
    'blenders_j1', 'blenders_j2', 'blenders_j3', 'blenders_j4',
    'demands_p1', 'demands_p2',
    'properties_j1_q1', 'properties_j2_q1', 'properties_j3_q1', 'properties_j4_q1',
    'sources_avail_next_0_s1', 'sources_avail_next_0_s2', 'demands_avail_next_0_p1', 'demands_avail_next_0_p2',
    'sources_avail_next_1_s1', 'sources_avail_next_1_s2', 'demands_avail_next_1_p1', 'demands_avail_next_1_p2',
    'sources_avail_next_2_s1', 'sources_avail_next_2_s2', 'demands_avail_next_2_p1', 'demands_avail_next_2_p2',
    'sources_avail_next_3_s1', 'sources_avail_next_3_s2', 'demands_avail_next_3_p1', 'demands_avail_next_3_p2',
    'sources_avail_next_4_s1', 'sources_avail_next_4_s2', 'demands_avail_next_4_p1', 'demands_avail_next_4_p2',
    'sources_avail_next_5_s1', 'sources_avail_next_5_s2', 'demands_avail_next_5_p1', 'demands_avail_next_5_p2',
    't'
]
cols_act = [
    'source_blend_s1_j1', 'source_blend_s1_j2', 'source_blend_s1_j3', 'source_blend_s1_j4',
    'source_blend_s2_j1', 'source_blend_s2_j2', 'source_blend_s2_j3', 'source_blend_s2_j4',
    'blend_demand_j1_p1', 'blend_demand_j1_p2',
    'blend_demand_j2_p1', 'blend_demand_j2_p2',
    'blend_demand_j3_p1', 'blend_demand_j3_p2',
    'blend_demand_j4_p1', 'blend_demand_j4_p2',
    'tau_s1', 'tau_s2',
    'delta_p1', 'delta_p2'
]

dataobs, dataact = [], []

now = datetime.datetime.now().strftime('%m%d-%H%M')

for n in range(N):
    total_product = rd.randint(100,200)
    th.rand(12)
    sources = th.rand(10)
    demands = th.rand(10)
    
    sources = sources.softmax(dim = 0) * total_product
    demands = demands.softmax(dim = 0) * total_product
    
    sources = th.nn.functional.dropout(sources, p=0.2)
    demands = th.nn.functional.dropout(demands, p=0.2)
    
    s_amounts = {'s1': [sources[0].item().__round__(1), sources[1].item().__round__(1), sources[2].item().__round__(1), sources[3].item().__round__(1), sources[4].item().__round__(1), 0], 
                 's2': [sources[5].item().__round__(1), sources[6].item().__round__(1), sources[7].item().__round__(1), sources[8].item().__round__(1), sources[9].item().__round__(1), 0]}
    d_amounts = {'d1': [0, demands[0].item().__round__(1), demands[1].item().__round__(1), demands[2].item().__round__(1), demands[3].item().__round__(1), demands[4].item().__round__(1)], 
                 'd2': [0, demands[5].item().__round__(1), demands[6].item().__round__(1), demands[7].item().__round__(1), demands[8].item().__round__(1), demands[9].item().__round__(1)]}

    r = rd.randint(total_product//5, total_product//4)
    b_inv_lb = {b: 0 for b in blenders}
    b_inv_ub = {b: r for b in blenders}

    print("\n\n\ntotal_product:", total_product, "\ns_amounts:", s_amounts, "\nd_amounts:", d_amounts, "\nb_inv_lb:", b_inv_lb, "\nb_inv_ub:", b_inv_ub)
    
    continue

    # Model
    model = ConcreteModel()

    # Sets
    model.sources = Set(initialize=sources)
    model.demands = Set(initialize=demands)
    model.blenders = Set(initialize=blenders)
    model.properties = Set(initialize=properties)
    model.timestamps = Set(initialize=timestamps)

    # Parameters
    model.s_inv_lb = Param(model.sources, initialize=s_inv_lb)
    model.s_inv_ub = Param(model.sources, initialize=s_inv_ub)
    model.s_amounts = Param(model.sources, initialize=s_amounts)
    model.d_quals_lb = Param(model.demands, initialize=d_quals_lb)
    model.d_quals_ub = Param(model.demands, initialize=d_quals_ub)
    model.d_inv_lb = Param(model.demands, initialize=d_inv_lb)
    model.d_inv_ub = Param(model.demands, initialize=d_inv_ub)
    model.d_amounts = Param(model.demands, initialize=d_amounts)
    model.betaT_d = Param(model.demands, initialize=betaT_d)
    model.b_inv_ub = Param(model.blenders, initialize=b_inv_ub)

    # Decision variables
    model.source_inv = Var(model.sources, model.timestamps, domain=NonNegativeReals)
    model.blend_inv = Var(model.blenders, model.timestamps, domain=NonNegativeReals)
    model.demand_inv = Var(model.demands, model.timestamps, domain=NonNegativeReals)
    model.demand_inv_after_sell = Var(model.demands, model.timestamps, domain=NonNegativeReals)

    model.demand_sold = Var(model.demands, model.timestamps, domain=NonNegativeReals) # Represents the amount of product sold at each timestep; necessary for objective function

    model.prop_blend_inv = Var(model.properties, model.blenders, model.timestamps, domain=NonNegativeReals)

    model.source_blend_flow = Var(model.sources, model.blenders, model.timestamps, domain=NonNegativeReals)
    # model.source_demand_flow = Var(model.sources, model.demands, model.timestamps, domain=NonNegativeReals)
    model.blend_blend_flow = Var(model.blenders, model.blenders, model.timestamps, domain=NonNegativeReals)
    model.blend_demand_flow = Var(model.blenders, model.demands, model.timestamps, domain=NonNegativeReals)

    model.source_blend_bin = Var(model.sources, model.blenders, model.timestamps, domain=Binary)
    # model.source_demand_bin = Var(model.sources, model.demands, model.timestamps, domain=Binary)
    model.blend_blend_bin = Var(model.blenders, model.blenders, model.timestamps, domain=Binary)
    model.blend_demand_bin = Var(model.blenders, model.demands, model.timestamps, domain=Binary)

    # flow = 0 if the pair is not in the dict connections
    def connections_rule0_1(model, s, j, t):
        if j not in connections["source_blend"][s]:
            return model.source_blend_flow[s, j, t] == 0
        else:
            return model.source_blend_flow[s, j, t] >= 0

    # def connections_rule0_2(model, s, p, t):
    #     if p not in connections["source_demand"][s]:
    #         return model.source_demand_flow[s, p, t] == 0
        
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

    def material_balance_rule1(model, s, t):
        if t == 0:
            return model.source_inv[s, t] == model.s_amounts[s][t] \
                                            - sum(model.source_blend_flow[s, j, t] for j in model.blenders) # Initialize inventory at t=0
        else:
            return model.source_inv[s, t] == model.source_inv[s, model.timestamps.prev(t)] \
                                            + model.s_amounts[s][t] \
                                            - sum(model.source_blend_flow[s, j, t] for j in model.blenders)

    model.material_balance_rule1 = Constraint(model.sources, model.timestamps, rule=material_balance_rule1)

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
        return model.demand_sold[p, t] <= model.d_amounts[p][t]

    # Cannot sell more than what is available
    def material_balance_rule3_1(model, p, t):
        return model.demand_sold[p, t] <= model.demand_inv[p, t]

    def material_balance_rule3_2(model, p, t):
        if t == 0:
            return model.demand_inv[p, t] == 0 # Initialize inventory at t=0
        else:
            return model.demand_inv[p, t] == model.demand_inv_after_sell[p, model.timestamps.prev(t)] \
                                        + sum(model.blend_demand_flow[j, p, t] for j in model.blenders)
                                        
    def material_balance_rule3_3(model, p, t):
        if t == 0:
            return model.demand_inv_after_sell[p, t] == 0 # Initialize inventory at t=0
        else:
            return model.demand_inv_after_sell[p, t] == model.demand_inv[p, t] - model.demand_sold[p, t] 

    model.material_balance_rule3_0 = Constraint(model.demands, model.timestamps, rule=material_balance_rule3_0)
    model.material_balance_rule3_1 = Constraint(model.demands, model.timestamps, rule=material_balance_rule3_1)
    model.material_balance_rule3_2 = Constraint(model.demands, model.timestamps, rule=material_balance_rule3_2)
    model.material_balance_rule3_3 = Constraint(model.demands, model.timestamps, rule=material_balance_rule3_3)

    def material_balance_rule3_2(model, p, t):
        if t == 0:
            return model.demand_inv[p, t] == 0 # Initialize inventory at t=0
        else:
            return model.demand_inv[p, t] == model.demand_inv[p, model.timestamps.prev(t)] \
                                        + sum(model.blend_demand_flow[j, p, t] for j in model.blenders) \
                                        - model.demand_sold[p, t] 

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

    solver.options['timelimit'] = 2*60

    result = solver.solve(model, tee=False)
    solveinfo = result.json_repn()
    
    
    #### Appending data ####
    
    
    for t in range(T):
        dataobs.append([
            model.source_inv["s1", t].value,                #  (0, ['sources', 's1']),
            model.source_inv["s2", t].value,                #  (1, ['sources', 's2']),
            model.blend_inv["b1", t].value,                 #  (2, ['blenders', 'j1']),
            model.blend_inv["b2", t].value,                 #  (3, ['blenders', 'j2']),
            model.blend_inv["b3", t].value,                 #  (4, ['blenders', 'j3']),
            model.blend_inv["b4", t].value,                 #  (5, ['blenders', 'j4']),
            model.demand_inv_after_sell["d1", t].value,     #  (6, ['demands', 'p1']),
            model.demand_inv_after_sell["d2", t].value,     #  (7, ['demands', 'p2']),
            model.prop_blend_inv["p1", "b1", t].value,      #  (8, ['properties', 'j1', 'q1']),
            model.prop_blend_inv["p1", "b2", t].value,      #  (9, ['properties', 'j2', 'q1']),
            model.prop_blend_inv["p1", "b3", t].value,      #  (10, ['properties', 'j3', 'q1']),
            model.prop_blend_inv["p1", "b4", t].value,      #  (11, ['properties', 'j4', 'q1']),
            
            s_amounts["s1"][0+t] if 0+t < T else 0,  #  (12, ['sources_avail_next_0', 's1']),
            s_amounts["s2"][0+t] if 0+t < T else 0,  #  (13, ['sources_avail_next_0', 's2']),
            d_amounts["d1"][0+t] if 0+t < T else 0,  #  (14, ['demands_avail_next_0', 'p1']),
            d_amounts["d2"][0+t] if 0+t < T else 0,  #  (15, ['demands_avail_next_0', 'p2']),
            
            s_amounts["s1"][1+t] if 1+t < T else 0,  #  (16, ['sources_avail_next_1', 's1']),
            s_amounts["s2"][1+t] if 1+t < T else 0,  #  (17, ['sources_avail_next_1', 's2']),
            d_amounts["d1"][1+t] if 1+t < T else 0,  #  (18, ['demands_avail_next_1', 'p1']),
            d_amounts["d2"][1+t] if 1+t < T else 0,  #  (19, ['demands_avail_next_1', 'p2']),
            
            s_amounts["s1"][2+t] if 2+t < T else 0,  #  (20, ['sources_avail_next_2', 's1']),
            s_amounts["s2"][2+t] if 2+t < T else 0,  #  (21, ['sources_avail_next_2', 's2']),
            d_amounts["d1"][2+t] if 2+t < T else 0,  #  (22, ['demands_avail_next_2', 'p1']),
            d_amounts["d2"][2+t] if 2+t < T else 0,  #  (23, ['demands_avail_next_2', 'p2']),
            
            s_amounts["s1"][3+t] if 3+t < T else 0,  #  (24, ['sources_avail_next_3', 's1']),
            s_amounts["s2"][3+t] if 3+t < T else 0,  #  (25, ['sources_avail_next_3', 's2']),
            d_amounts["d1"][3+t] if 3+t < T else 0,  #  (26, ['demands_avail_next_3', 'p1']),
            d_amounts["d2"][3+t] if 3+t < T else 0,  #  (27, ['demands_avail_next_3', 'p2']),
            
            s_amounts["s1"][4+t] if 4+t < T else 0,  #  (28, ['sources_avail_next_4', 's1']),
            s_amounts["s2"][4+t] if 4+t < T else 0,  #  (29, ['sources_avail_next_4', 's2']),
            d_amounts["d1"][4+t] if 4+t < T else 0,  #  (30, ['demands_avail_next_4', 'p1']),
            d_amounts["d2"][4+t] if 4+t < T else 0,  #  (31, ['demands_avail_next_4', 'p2']),
            
            s_amounts["s1"][5+t] if 5+t < T else 0,  #  (32, ['sources_avail_next_5', 's1']),
            s_amounts["s2"][5+t] if 5+t < T else 0,  #  (33, ['sources_avail_next_5', 's2']),
            d_amounts["d1"][5+t] if 5+t < T else 0,  #  (34, ['demands_avail_next_5', 'p1']),
            d_amounts["d2"][5+t] if 5+t < T else 0,  #  (35, ['demands_avail_next_5', 'p2']),
            t                                        #  (36, ['t'])
        ])
        
        dataact.append([
            model.source_blend_flow["s1", "b1", t].value,  # (0, ['source_blend', 's1', 'j1']),
            model.source_blend_flow["s1", "b2", t].value,  # (1, ['source_blend', 's1', 'j2']),
            model.source_blend_flow["s1", "b3", t].value,  # (2, ['source_blend', 's1', 'j3']),
            model.source_blend_flow["s1", "b4", t].value,  # (3, ['source_blend', 's1', 'j4']),
            model.source_blend_flow["s2", "b1", t].value,  # (4, ['source_blend', 's2', 'j1']),
            model.source_blend_flow["s2", "b2", t].value,  # (5, ['source_blend', 's2', 'j2']),
            model.source_blend_flow["s2", "b3", t].value,  # (6, ['source_blend', 's2', 'j3']),
            model.source_blend_flow["s2", "b4", t].value,  # (7, ['source_blend', 's2', 'j4']),
            
            model.blend_demand_flow["b1", "d1", t].value,  # (8, ['blend_demand', 'j1', 'p1']),
            model.blend_demand_flow["b1", "d2", t].value,  # (9, ['blend_demand', 'j1', 'p2']),
            model.blend_demand_flow["b2", "d1", t].value,  # (10, ['blend_demand', 'j2', 'p1']),
            model.blend_demand_flow["b2", "d2", t].value,  # (11, ['blend_demand', 'j2', 'p2']),
            model.blend_demand_flow["b3", "d1", t].value,  # (12, ['blend_demand', 'j3', 'p1']),
            model.blend_demand_flow["b3", "d2", t].value,  # (13, ['blend_demand', 'j3', 'p2']),
            model.blend_demand_flow["b4", "d1", t].value,  # (14, ['blend_demand', 'j4', 'p1']),
            model.blend_demand_flow["b4", "d2", t].value,  # (15, ['blend_demand', 'j4', 'p2']),
            
            model.offer_bought["s1", t].value,  # (16, ['tau', 's1']),
            model.offer_bought["s2", t].value,  # (17, ['tau', 's2']),
            model.demand_sold["d1", t].value,   # (18, ['delta', 'p1']),
            model.demand_sold["d2", t].value,   # (19, ['delta', 'p2'])
        ])
        
    print(f"{n}/{N}\t available product: {total_product}\tOptimal value: {solveinfo['Problem']['Lower bound'].__round__(5)}\tExit code: {solveinfo['Solver']['Status']}")


