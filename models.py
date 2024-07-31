import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import *
from stable_baselines3.common.callbacks import *
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3 import PPO
from gymnasium import spaces 



# Utilities

def lr_scheduler(p):
    if p > 0.9:
        return 4e-2
    if p > 0.75:
        return 2e-2
    if p > 0.4:
        return 5e-3
    else:
        return 1e-3
    
def lr_scheduler_mult(p):
    if p > 0.9:
        return 4e-4
    if p > 0.75:
        return 2e-4
    if p > 0.4:
        return 1e-4
    else:
        return 5e-5

def replace_columns(X: th.Tensor, Y: th.Tensor, indices: list):
    """
        Replaces X's indices[i]-th column with Y's i-th column
    """
    X_new = X.clone()
    
    for i, idx in enumerate(indices):
        X_new[:, idx] = Y[:, i]
    
    return X_new




# Modules

class CustomMLPPolicy(nn.Module):
    def __init__(self,
                features_dim,
                dims_pi = [64, 64],
                dims_vf = [64, 64],
                act_pi_cls = nn.ReLU,
                act_vf_cls = nn.ReLU,
                *args,
                **kwargs):
        super().__init__()

        layers_pi, layers_vf = [], []
        dims_pi = [features_dim] + dims_pi
        dims_vf = [features_dim] + dims_vf
        
        for k in range(len(dims_pi)-1):
            layers_pi.append(nn.Linear(in_features = dims_pi[k], out_features = dims_pi[k+1]))
            layers_pi.append(act_pi_cls())
        
        for k in range(len(dims_vf)-1):
            layers_vf.append(nn.Linear(in_features = dims_vf[k], out_features = dims_vf[k+1]))
            layers_vf.append(act_vf_cls())
            
        self.latent_dim_pi = dims_pi[-1]
        self.latent_dim_vf = dims_vf[-1]
            
        self.model_pi = nn.Sequential(*layers_pi)
        self.model_vf = nn.Sequential(*layers_vf)
    
    def forward(self, features: th.Tensor):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return(self.model_pi(features))
        
    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return(self.model_vf(features))
    
class CustomMLPPolicyTanh(CustomMLPPolicy):
    def __init__(self, features_dim, maxflow = 50, dims_pi = [64, 64], dims_vf = [64, 64], act_pi_cls = nn.ReLU, act_vf_cls = nn.ReLU, *args, **kwargs):
        super().__init__(features_dim, dims_pi = dims_pi, dims_vf = dims_vf, act_pi_cls = act_pi_cls, act_vf_cls = act_vf_cls, *args, **kwargs)
        
        self.maxflow = maxflow
        print("maxflow:", maxflow)
        
        layers_pi = []
        dims_pi = [features_dim] + dims_pi
        
        for k in range(len(dims_pi)-1):
            layers_pi.append(nn.Linear(in_features = dims_pi[k], out_features = dims_pi[k+1]))
            layers_pi.append(act_pi_cls())
        
        layers_pi.append(nn.Tanh())
        
        self.latent_dim_pi = dims_pi[-1]
            
        self.model_pi = nn.Sequential(*layers_pi)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return(self.model_pi(features) * self.maxflow)

class CustomRNNPolicy(nn.Module):
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
        num_layers_pi = 2,
        num_layers_vf = 2,
        *args,
        **kwargs
    ):
        super().__init__()

        self.features_dim = features_dim
        
        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.num_layers_pi = num_layers_pi
        self.num_layers_vf = num_layers_vf
        
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        
        self.hidden_size_pi = hidden_size_pi
        self.hidden_size_vf = hidden_size_vf
        
        self.seq_pi = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size_pi, output_size_pi),
            nn.ReLU(),
            nn.Linear(output_size_pi, 2*output_size_pi),
            nn.ReLU(),
            nn.Linear(2*output_size_pi, output_size_pi),
            nn.ReLU()
        )
        self.lstm_pi = nn.LSTM(input_size=1, hidden_size=hidden_size_pi, num_layers=self.num_layers_pi, batch_first=True)
        
        self.seq_vf = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size_vf, output_size_vf),
            nn.ReLU(),
            nn.Linear(output_size_vf, 2*output_size_vf),
            nn.ReLU(),
            nn.Linear(2*output_size_vf, output_size_vf),
            nn.ReLU()
        )
        self.lstm_vf = nn.LSTM(input_size=1, hidden_size=hidden_size_vf, num_layers=self.num_layers_vf, batch_first=True)

    def forward(self, features: th.Tensor):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        
        batch_size = features.size(0)
        features = features.unsqueeze(0)
        features = features.reshape(batch_size, self.features_dim, -1)
        
        h0_pi = th.zeros(self.num_layers_pi, batch_size, self.hidden_size_pi).to(features.device).detach()
        c0_pi = th.zeros(self.num_layers_pi, batch_size, self.hidden_size_pi).to(features.device).detach()

        lstm_out, _ = self.lstm_pi(features, (h0_pi, c0_pi))

        out = self.seq_pi(lstm_out[:, -1, :])
        return out

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        batch_size = features.size(0)
        features = features.unsqueeze(0)
        features = features.reshape(batch_size, self.features_dim, 1)
        
        h0_vf = th.zeros(self.num_layers_vf, batch_size, self.hidden_size_vf).to(features.device).detach()
        c0_vf = th.zeros(self.num_layers_vf, batch_size, self.hidden_size_vf).to(features.device).detach()

        lstm_out, _ = self.lstm_vf(features, (h0_vf, c0_vf))

        out = self.seq_vf(lstm_out[:, -1, :])
        return out



# Policies
    
class CustomRNN_ACP(ActorCriticPolicy):
    def _build_mlp_extractor(self):
        self.mlp_extractor = CustomRNNPolicy(self.features_dim)

class CustomMLP_ACP(ActorCriticPolicy):
    def _build_mlp_extractor(self):
        self.mlp_extractor = CustomMLPPolicy(self.features_dim)
        
class CustomMLP_ACP_simplest_tanh(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule: Callable[[float], float], net_arch = None, activation_fn: nn.Module = nn.Tanh, ortho_init: bool = True, use_sde: bool = False, log_std_init: float = 0, full_std: bool = True, use_expln: bool = False, squash_output: bool = False, features_extractor_class = FlattenExtractor, features_extractor_kwargs = None, share_features_extractor: bool = True, normalize_images: bool = True, optimizer_class = th.optim.Adam, optimizer_kwargs = None):
        super().__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn, ortho_init, use_sde, log_std_init, full_std, use_expln, squash_output, features_extractor_class, features_extractor_kwargs, share_features_extractor, normalize_images, optimizer_class, optimizer_kwargs)
        
        self.action_dist = make_my_proba_distribution(action_space, use_squash=True)
        self._build(lr_schedule)
        
    def _build_mlp_extractor(self):
        self.mlp_extractor = CustomMLPPolicy(self.features_dim)
        
    def forward(self, obs: th.Tensor, deterministic: bool = False):
        # Make log std fixed ?
        actions, values, log_prob = super().forward(obs, deterministic)
        
        print("before:", actions)
        # actions = nn.Tanh()(actions) * self.observation_space.high[0]
        # print("after:", actions)
        
        # self.log_std = th.clip(self.log_std, 0, self.log_std_init + 2)
        return actions, values, log_prob

class CustomMLP_ACP_simplest_std(ActorCriticPolicy):
    def _build_mlp_extractor(self):
        self.mlp_extractor = CustomMLPPolicy(self.features_dim, dims_pi=[128]*4, dims_vf=[128]*4)
        
    def forward(self, obs: th.Tensor, deterministic: bool = False):
        # Make log std fixed ?
        actions, values, log_prob = super().forward(obs, deterministic)
        with th.no_grad():
            self.log_std.copy_(th.clip(self.log_std, -999, 2))
        
        return actions, values, log_prob
    
class CustomMLP_ACP_simplest_softmax(ActorCriticPolicy):
    def _build_mlp_extractor(self):
        self.mlp_extractor = CustomMLPPolicy(self.features_dim, dims_pi=[128]*4, dims_vf=[128]*4)
        
    def forward(self, obs: th.Tensor, deterministic: bool = False):
        # Make log std fixed ?
        actions, values, log_prob = super().forward(obs, deterministic)
        
        print("\nbefore:", actions)
        
        M = 20
        actions = actions.relu()
        idx = [0, 1] # indices of in-flow and out-flow of the only blending tank ; Should be list of pairs of indices: [([0,1,2], [3,4,5]) , (...)]
        in_out = actions[:, idx]
        exp_actions = th.exp(M * in_out)
        softmax_actions = exp_actions / exp_actions.sum(dim=1, keepdim=True)
        actions_masked = (in_out * softmax_actions).nan_to_num()

        actions = replace_columns(actions, actions_masked, idx)
        
        print("after:", actions)
        
        # self.log_std = th.clip(self.log_std, 0, self.log_std_init + 2)
        return actions, values, log_prob



# Algorithms

class CustomPPO(PPO):
    def train(self):
        super().train()


def make_my_proba_distribution(
    action_space: spaces.Space, use_sde: bool = False, use_squash: bool = False, dist_kwargs: Optional[Dict[str, Any]] = None
) -> Distribution:
    """
    Return an instance of Distribution for the correct type of action space

    :param action_space: the input action space
    :param use_sde: Force the use of StateDependentNoiseDistribution
        instead of DiagGaussianDistribution
    :param dist_kwargs: Keyword arguments to pass to the probability distribution
    :return: the appropriate Distribution object
    """
    if dist_kwargs is None:
        dist_kwargs = {}

    if isinstance(action_space, spaces.Box):
        if use_sde: 
            cls = StateDependentNoiseDistribution 
        elif use_squash: 
            cls = SquashedDiagGaussianDistribution
        else: 
            cls = DiagGaussianDistribution
        return cls(get_action_dim(action_space), **dist_kwargs)
    elif isinstance(action_space, spaces.Discrete):
        return CategoricalDistribution(int(action_space.n), **dist_kwargs)
    elif isinstance(action_space, spaces.MultiDiscrete):
        return MultiCategoricalDistribution(list(action_space.nvec), **dist_kwargs)
    elif isinstance(action_space, spaces.MultiBinary):
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return BernoulliDistribution(action_space.n, **dist_kwargs)
    else:
        raise NotImplementedError(
            "Error: probability distribution, not implemented for action space"
            f"of type {type(action_space)}."
            " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary."
        )



# Imitation learning models

class IL_MLP_simple(nn.Module):
    def __init__(self,
                dims = [64, 64],
                act_cls = nn.ReLU,
                *args,
                **kwargs):
        super().__init__()

        IN_DIM, OUT_DIM = 37, 20

        layers = []
        dims = [IN_DIM] + dims + [OUT_DIM]
        
        for k in range(len(dims)-1):
            layers.append(nn.Linear(in_features = dims[k], out_features = dims[k+1]))
            layers.append(act_cls())
        
        self.latent_dim_pi = dims[-1]
            
        self.model_pi = nn.Sequential(*layers)
    
    def forward(self, x: th.Tensor):
        x = self.model_pi(x)
        
        # x = 50 * th.nn.functional.tanh(x)
        
        return x

if __name__ == "__main__":
    # model = PPO(CustomMLP_ACP_simplest, "CartPole-v1", verbose=1)
    
    from envs import simplestenv
    model = PPO(CustomMLP_ACP_simplest_std, simplestenv, verbose=1)
    model.learn(200000)

