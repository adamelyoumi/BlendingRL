import torch as th

def get_bin(n):
    return f"{(n//12)*12+1}-{(n//12+1)*12}"

def get_sbp(connections):
    # Function to obtain the list of source, blending, demand tank names from the connections
    
    sources = list(connections["source_blend"].keys())
    
    b_list = list(connections["blend_blend"].keys())
    for b in connections["blend_blend"].keys():
        b_list += connections["blend_blend"][b]
    b_list += list(connections["blend_demand"].keys())
    blenders = list(set(b_list))
    
    p_list = []
    for p in connections["blend_demand"].keys():
        p_list += connections["blend_demand"][p]
    demands = list(set(p_list))
    
    return sources, blenders, demands

def get_jsons(layout):
    import json
    with open(f"./configs/json/connections_{layout}.json" ,"r") as f:
        connections_s = f.readline()
    connections = json.loads(connections_s)

    with open(f"./configs/json/action_sample_{layout}.json" ,"r") as f:
        action_sample_s = f.readline()
    action_sample = json.loads(action_sample_s)
    return connections, action_sample


def get_observations(model):
    dataobs = []
    T = len(model.timestamps.data())
    for t in range(T):
        dataobs.append([
            model.source_inv["s1", t].value,                #  (0, ['sources', 's1']),
            model.source_inv["s2", t].value,                #  (1, ['sources', 's2']),
            model.blend_inv["j1", t].value,                 #  (2, ['blenders', 'j1']),
            model.blend_inv["j2", t].value,                 #  (3, ['blenders', 'j2']),
            model.blend_inv["j3", t].value,                 #  (4, ['blenders', 'j3']),
            model.blend_inv["j4", t].value,                 #  (5, ['blenders', 'j4']),
            model.demand_inv_after_sell["p1", t].value,     #  (6, ['demands', 'p1']),
            model.demand_inv_after_sell["p2", t].value,     #  (7, ['demands', 'p2']),
            model.prop_blend_inv["q1", "j1", t].value,      #  (8, ['properties', 'j1', 'q1']),
            model.prop_blend_inv["q1", "j2", t].value,      #  (9, ['properties', 'j2', 'q1']),
            model.prop_blend_inv["q1", "j3", t].value,      #  (10, ['properties', 'j3', 'q1']),
            model.prop_blend_inv["q1", "j4", t].value,      #  (11, ['properties', 'j4', 'q1']),
            
            model.tau0["s1"][0+t] if 0+t < T else 0,        #  (12, ['sources_avail_next_0', 's1']),
            model.tau0["s2"][0+t] if 0+t < T else 0,        #  (13, ['sources_avail_next_0', 's2']),
            model.delta0["p1"][0+t] if 0+t < T else 0,      #  (14, ['demands_avail_next_0', 'p1']),
            model.delta0["p2"][0+t] if 0+t < T else 0,      #  (15, ['demands_avail_next_0', 'p2']),

            model.tau0["s1"][1+t] if 1+t < T else 0,        #  (16, ['sources_avail_next_1', 's1']),
            model.tau0["s2"][1+t] if 1+t < T else 0,        #  (17, ['sources_avail_next_1', 's2']),
            model.delta0["p1"][1+t] if 1+t < T else 0,      #  (18, ['demands_avail_next_1', 'p1']),
            model.delta0["p2"][1+t] if 1+t < T else 0,      #  (19, ['demands_avail_next_1', 'p2']),

            model.tau0["s1"][2+t] if 2+t < T else 0,        #  (20, ['sources_avail_next_2', 's1']),
            model.tau0["s2"][2+t] if 2+t < T else 0,        #  (21, ['sources_avail_next_2', 's2']),
            model.delta0["p1"][2+t] if 2+t < T else 0,      #  (22, ['demands_avail_next_2', 'q1']),
            model.delta0["p2"][2+t] if 2+t < T else 0,      #  (23, ['demands_avail_next_2', 'p2']),

            model.tau0["s1"][3+t] if 3+t < T else 0,        #  (24, ['sources_avail_next_3', 's1']),
            model.tau0["s2"][3+t] if 3+t < T else 0,        #  (25, ['sources_avail_next_3', 's2']),
            model.delta0["p1"][3+t] if 3+t < T else 0,      #  (26, ['demands_avail_next_3', 'q1']),
            model.delta0["p2"][3+t] if 3+t < T else 0,      #  (27, ['demands_avail_next_3', 'p2']),

            model.tau0["s1"][4+t] if 4+t < T else 0,        #  (28, ['sources_avail_next_4', 's1']),
            model.tau0["s2"][4+t] if 4+t < T else 0,        #  (29, ['sources_avail_next_4', 's2']),
            model.delta0["p1"][4+t] if 4+t < T else 0,      #  (30, ['demands_avail_next_4', 'q1']),
            model.delta0["p2"][4+t] if 4+t < T else 0,      #  (31, ['demands_avail_next_4', 'p2']),

            model.tau0["s1"][5+t] if 5+t < T else 0,        #  (32, ['sources_avail_next_5', 's1']),
            model.tau0["s2"][5+t] if 5+t < T else 0,        #  (33, ['sources_avail_next_5', 's2']),
            model.delta0["p1"][5+t] if 5+t < T else 0,      #  (34, ['demands_avail_next_5', 'q1']),
            model.delta0["p2"][5+t] if 5+t < T else 0,      #  (35, ['demands_avail_next_5', 'p2']),
            t                                               #  (36, ['t'])
        ])
    return th.Tensor(dataobs)

def get_actions(model):
    dataact = []
    for t in range(len(model.timestamps.data())):
        dataact.append([
                    model.source_blend_flow["s1", "j1", t].value,  # (0, ['source_blend', 's1', 'j1']),
                    model.source_blend_flow["s1", "j2", t].value,  # (1, ['source_blend', 's1', 'j2']),
                    model.source_blend_flow["s1", "j3", t].value,  # (2, ['source_blend', 's1', 'j3']),
                    model.source_blend_flow["s1", "j4", t].value,  # (3, ['source_blend', 's1', 'j4']),
                    model.source_blend_flow["s2", "j1", t].value,  # (4, ['source_blend', 's2', 'j1']),
                    model.source_blend_flow["s2", "j2", t].value,  # (5, ['source_blend', 's2', 'j2']),
                    model.source_blend_flow["s2", "j3", t].value,  # (6, ['source_blend', 's2', 'j3']),
                    model.source_blend_flow["s2", "j4", t].value,  # (7, ['source_blend', 's2', 'j4']),
                    
                    model.blend_demand_flow["j1", "p1", t].value,  # (8, ['blend_demand', 'j1', 'p1']),
                    model.blend_demand_flow["j1", "p2", t].value,  # (9, ['blend_demand', 'j1', 'p2']),
                    model.blend_demand_flow["j2", "p1", t].value,  # (10, ['blend_demand', 'j2', 'p1']),
                    model.blend_demand_flow["j2", "p2", t].value,  # (11, ['blend_demand', 'j2', 'p2']),
                    model.blend_demand_flow["j3", "p1", t].value,  # (12, ['blend_demand', 'j3', 'p1']),
                    model.blend_demand_flow["j3", "p2", t].value,  # (13, ['blend_demand', 'j3', 'p2']),
                    model.blend_demand_flow["j4", "p1", t].value,  # (14, ['blend_demand', 'j4', 'p1']),
                    model.blend_demand_flow["j4", "p2", t].value,  # (15, ['blend_demand', 'j4', 'p2']),
                    
                    model.offer_bought["s1", t].value,              # (16, ['tau', 's1']),
                    model.offer_bought["s2", t].value,              # (17, ['tau', 's2']),
                    model.demand_sold["p1", t].value,               # (18, ['delta', 'p1']),
                    model.demand_sold["p2", t].value,               # (19, ['delta', 'p2'])
        ])
    return th.Tensor(dataact)