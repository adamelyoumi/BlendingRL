import numpy as np
import pandas as pd
import random as rd
import datetime, time
import math as m
import argparse
import json, os, sys

from pyomo.environ import *
from utils import *

def solve(tau0, delta0, layout):
    with open(f"./configs/json/connections_{layout}.json" ,"r") as f:
        connections_s = f.readline()
    connections = json.loads(connections_s)
    
    sources, blenders, demands = get_sbp(connections)
    
    T = 6
    alpha = 0.1
    beta = 0.1
    
    properties = ["p1"]
    timestamps = list(range(T))

    sigma = {"s1":{"p1": 0.06}, "s2":{"p1": 0.26}}

    sigma_ub = {"d1":{"p1": 0.16}, "d2":{"p1": 1}}
    sigma_lb = {"d1":{"p1": 0}, "d2":{"p1": 0}}

    s_inv_ub = {'s1': 30, 's2': 30}

    d_quals_lb = {'d1': 0, 'd2': 0}
    d_quals_ub = {'d1': 0.16, 'd2': 0.1}

    d_inv_ub = {'d1': 30, 'd2': 30}
    betaT_d = {'d1': 2, 'd2': 1}
    ...





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('N')
    args = parser.parse_args() 
    N = int(args.N)
