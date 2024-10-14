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

