# BlendingRL
Deep Learning and RL Approach to the multiperiod Blending problem

## Example environment
![image info](img/env_base.png)

*Source: [An MILP-MINLP decomposition method for the global optimization of a source based model of the multiperiod blending problem](https://optimization-online.org/wp-content/uploads/2015/04/4864.pdf)*

Our environment is comprised of source tanks that generate product according to supply values at each timestep, blending tanks that take product from source and/or other blending tanks, and demand tanks that take product from blending tanks and sell the product depending on the concentrations and the demand values at the given timestep.

A critical constraint is that no blending tank can have both incoming and outgoing flow at the same timestep. The environment are coded to prevent and penalize illegal actions that disrespect this rule, as well as other "common sense rules" (no action that would result in negative inventories after each timestep, etc.)

We use pytorch, stable_baselines3 and Tensorboard for visualization.

Please check out the PDF file for the MDP definition, `envs.py` for the environment definition, `models.py` for custom policy and model definitions, and the `RL_scripts/RL_train.ipynb` notebook for the most recent training process.

Training logs can be viewed by opening tensorboard on the `logs` directory. Configuration descriptions can be found in the `configs` directory or in [this Google Sheet](https://docs.google.com/spreadsheets/d/1qYxyulHMTBfmBNzkTz9xo487xnKjwOy5LTgq75R-mkQ/edit?usp=sharing)

The current best configuration (ID 23) uses PPO with a scheduled variance, randomized supply/demand values at each episode, and non-zero reward incentives (D, Z parameters) and penalties (P, B, M)

Requirements file in `aws/reqs.txt`