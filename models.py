import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.distributions import DiagGaussianDistribution
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces



class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        features_dim: int,
        last_layer_dim_pi: int = 64,
        hidden_size_pi = 64,
        output_size_pi = 64,
        
        last_layer_dim_vf: int = 64,
        hidden_size_vf = 64,
        output_size_vf = 64,
    ):
        super().__init__()

        self.features_dim = features_dim
        
        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        
        self.hidden_size_pi = hidden_size_pi
        self.hidden_size_vf = hidden_size_vf
        
        self.seq_pi = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size_pi, output_size_pi),
            nn.ReLU()
        )
        self.lstm_pi = nn.LSTM(input_size=self.features_dim, hidden_size=hidden_size_pi, num_layers=1, batch_first=True)
        self.hidden_pi = None
        
        self.seq_vf = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size_vf, output_size_vf),
            nn.ReLU()
        )
        self.lstm_vf = nn.LSTM(input_size=self.features_dim, hidden_size=hidden_size_vf, num_layers=1, batch_first=True)
        self.hidden_vf = None

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        if self.hidden_pi is None:
            self.batch_size = features.size(0)
            self.hidden_pi = (th.zeros(self.batch_size, self.hidden_size_pi).to(features.device),
                            th.zeros(self.batch_size, self.hidden_size_pi).to(features.device))

        # print(self.batch_size, features, self.hidden_pi)
        lstm_out, self.hidden_pi = self.lstm_pi(features, self.hidden_pi)
        # Apply activation and linear layer to the output of the last LSTM time step
        out = self.seq_pi(lstm_out)
        
        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        if self.hidden_vf is None:
            self.batch_size = features.size(0)
            self.hidden_vf = (th.zeros(self.batch_size, self.hidden_size_vf).to(features.device),
                            th.zeros(self.batch_size, self.hidden_size_vf).to(features.device))

        # Forward pass through LSTM
        lstm_out, self.hidden_vf = self.lstm_vf(features, self.hidden_vf)

        # Apply activation and linear layer to the output of the last LSTM time step
        out = self.seq_vf(lstm_out)
        
        return out


class CustomActorCriticPolicy(ActorCriticPolicy):    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)


model = PPO(CustomActorCriticPolicy, "CartPole-v1", verbose=1)
model.learn(5000)
