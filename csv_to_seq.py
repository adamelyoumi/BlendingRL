# INPUT: CSV pair (action/observation) that contain one timestamp per row
# OUTPUT: List of dicts (1 elt per episode) that contains a dict with keys ['observations', 'actions', 'rewards', 'terminals'] 

import csv, json, os, sys
import pickle
from envs import BlendEnv
import torch as th
import numpy as np

M, Q, P, B, Z, D = 0, 0, 0, 0, 1, 0

with open("C:/Users/adame/OneDrive/Bureau/CODE/BlendingRL/configs/json/connections_simple.json" ,"r") as f:
    connections_s = f.readline()
connections = json.loads(connections_s)

with open("C:/Users/adame/OneDrive/Bureau/CODE/BlendingRL/configs/json/action_sample_simple.json" ,"r") as f:
    action = f.readline()
action_sample = json.loads(action)

def read_csv(file_path):
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader) # skip column names row
        return [row for row in reader]

def construct_episodes(observation_data, action_data):
    # cnt = 0
    # for p in (observation_data):
    #     print(cnt, p)
    #     cnt+=1
    episodes = []
    current_episode = {'observations': [], 
                       'next_observations': [], 
                       'actions': [], 
                       'rewards': [], 
                       'terminals': []}
    tau0   = {"s1": [], "s2": []}
    delta0 = {"p1": [], "p2": []}
    first_episode = True

    for obs, act in zip(observation_data, action_data):
        # print(obs)
        # print(np.array([float(i) for i in obs[1:]]))
        # obs, act = np.array([float(i) for i in obs[1:]]), np.array([float(i) for i in act[1:]])
        timestep = int(obs[-1])
        
        # If new episode start detected
        if timestep == 0 and not first_episode:
            
            # Computing rewards
            T = len(tau0["s1"])
            env = BlendEnv(M=M, Q=Q, P=P, B=B, Z=Z, D=D, tau0 = tau0, delta0 = delta0, T = T, connections = connections, action_sample = action_sample)
            obs_, _ = env.reset()
            prev_reward = 0
            for t in range(T):
                action = th.Tensor( [float(i) for i in current_episode["actions"][t]] )
                obs_, reward, done_, term_, _ = env.step(action)
                current_episode["rewards"].append(reward-prev_reward)
                prev_reward = reward
            # print([ob for ob in current_episode["observations"]])
            current_episode["observations"] = np.array(current_episode["observations"])
            current_episode["actions"] = np.array(current_episode["actions"])
            current_episode["rewards"] = np.array(current_episode["rewards"])
            
            # Adding ep to ep list
            episodes.append(current_episode)
            current_episode = {'observations': [],
                       'actions': [],
                       'rewards': [],
                       'terminals': []}
            tau0   = {"s1": [], "s2": []}
            delta0 = {"p1": [], "p2": []}

        # Append data
        # print(np.array([float(i) for i in obs[1:]]).shape, obs[1:], np.array([float(i) for i in obs[1:]]))
        current_episode["observations"].append( np.array([float(i) for i in obs[1:]]) )
        current_episode["actions"].append( np.array([float(i) for i in act[1:]]) )
        current_episode["terminals"].append( False )
        
        # print(current_episode["observations"][-1].shape, current_episode["observations"][-1])
        
        tau0["s1"].append(float(obs[13]))
        tau0["s2"].append(float(obs[14]))
        delta0["p1"].append(float(obs[15]))
        delta0["p2"].append(float(obs[16]))
        first_episode = False

    # Adding last episode of the csv
    T = len(tau0["s1"])
    env = BlendEnv(M=M, Q=Q, P=P, B=B, Z=Z, D=D, tau0 = tau0, delta0 = delta0, T = T, 
                    connections = connections, 
                    action_sample = action_sample)
    obs, _ = env.reset()
    prev_reward = 0
    for t in range(T):
        action = th.Tensor( [float(i) for i in current_episode["actions"][t]] )
        obs, reward, done, term, _ = env.step(action)
        current_episode["rewards"].append(reward-prev_reward)
        prev_reward = reward
    
    current_episode["observations"] = np.array(current_episode["observations"])
    current_episode["actions"] = np.array(current_episode["actions"])
    current_episode["rewards"] = np.array(current_episode["rewards"])
    
    # Adding ep to ep list
    episodes.append(current_episode)

    return episodes

def main(observation_csv, action_csv):
    observation_data = read_csv(observation_csv)
    action_data = read_csv(action_csv)

    episodes = construct_episodes(observation_data, action_data)

    return episodes

trajectories = []
for file in os.listdir("./data/simple/"):
    if "OBS" in file:
        obsfile = os.path.join("./data/simple/", file)
        actfile = os.path.join("./data/simple/", file.replace("OBS", "ACT"))
        print(obsfile)
        trajs_ = main(obsfile, actfile)
        print(len(trajs_))
        trajectories += trajs_

# Print the episodes to verify
# print(len(trajectories))
# for trajectory in trajectories:
#     print("New Episode")
#     for key in trajectory:
#         print(key, trajectory[key])

with open("./decision-transformer/gym/data/blend-small-v2.pkl", "wb") as f:
    pickle.dump(trajectories, f)