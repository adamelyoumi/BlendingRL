import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy #, register_policy
from stable_baselines3.common.distributions import DiagGaussianDistribution

class CustomNonNegativePolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(CustomNonNegativePolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[dict(vf=[64, 64], pi=[64, 64])],
            activation_fn=nn.ReLU,
            **kwargs
        )
        
        self.action_dist = DiagGaussianDistribution(2)

        self.action_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self.value_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        # self._initialize_policy()

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        action_logits = self.action_net(features)

        # Transform the action logits to enforce constraints
        x = th.relu(action_logits[:, 0])
        y = th.relu(action_logits[:, 1])

        # Ensure min(x, y) = 0
        actions = th.stack([x, th.zeros_like(y)], dim=1)
        actions[:, 1] = y * (x == 0).float()

        value = self.value_net(features)
        distribution = self.action_dist.proba_distribution(actions, self.log_std)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, value, log_prob

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        value = self.value_net(features)

        # Transform the actions to logits for the loss calculation
        action_logits = self.policy_net(features)
        x = th.relu(action_logits[:, 0])
        y = th.relu(action_logits[:, 1])
        transformed_actions = th.stack([x, th.zeros_like(y)], dim=1)
        transformed_actions[:, 1] = y * (x == 0).float()

        distribution = self.action_dist.proba_distribution(transformed_actions)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return value, log_prob, entropy

# Register the custom policy
# register_policy('CustomNonNegativePolicy', CustomNonNegativePolicy)
