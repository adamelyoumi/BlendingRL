from stable_baselines3 import A2C
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from policy_test import CustomNonNegativePolicy

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.observation_space = Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        self.action_space = Box(low=0, high=1, shape=(2,), dtype=np.float32)

    def reset(self, seed = 0):
        return (self.observation_space.sample(), {})

    def step(self, action):
        x, y = action
        # Example reward: penalize non-zero actions to demonstrate the transformation
        reward = - (x + y)
        done = False
        info = {}
        return self.observation_space.sample(), reward, False, done, info


# Create the environment
env = CustomEnv()

Y = A2C("MlpPolicy", env, verbose=1)
# Initialize the model with the custom policy
model = A2C(CustomNonNegativePolicy, env, verbose=1)

print(Y.policy)
print("\nwrong:", model.policy)

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("a2c_non_negative_policy")

# Load the model
model = A2C.load("a2c_non_negative_policy")

# Test the model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
