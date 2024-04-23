import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Space


class LSTMModel(nn.Module):

    def __init__(self, input_features, hidden_dim):
        super().__init__()
        self.input_features = input_features
        self.latent_dim_pi = hidden_dim
        self.latent_dim_vf = hidden_dim

        self.policy_lstm = nn.LSTM(input_features,
                                   hidden_dim,
                                   num_layers=3,
                                   batch_first=True)
        self.critic_lstm = nn.LSTM(input_features,
                                   hidden_dim,
                                   num_layers=3,
                                   batch_first=True)

        self.policy_fc = nn.Linear(3 * hidden_dim, hidden_dim)
        self.critic_fc = nn.Linear(3 * hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor):
        policy_lstm_output = self.policy_lstm(x)[1][0].permute(1, 0,
                                                               2).flatten(1)
        critic_lstm_output = self.critic_lstm(x)[1][0].permute(1, 0,
                                                               2).flatten(1)

        policy_lstm_output = F.relu(policy_lstm_output)
        critic_lstm_output = F.relu(critic_lstm_output)

        policy_output = self.policy_fc(policy_lstm_output)
        critic_output = self.critic_fc(critic_lstm_output)

        return policy_output, critic_output

    def forward_actor(self, x: torch.Tensor):
        policy_lstm_output = self.policy_lstm(x)[1][0].permute(1, 0,
                                                               2).flatten(1)

        policy_lstm_output = F.relu(policy_lstm_output)

        policy_output = self.policy_fc(policy_lstm_output)

        return policy_output

    def forward_critic(self, x: torch.Tensor):
        critic_lstm_output = self.critic_lstm(x)[1][0].permute(1, 0,
                                                               2).flatten(1)

        critic_lstm_output = F.relu(critic_lstm_output)

        critic_output = self.critic_fc(critic_lstm_output)

        return critic_output


class LSTMFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: Space) -> None:
        super().__init__(observation_space, observation_space.shape[1])

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations


class LSTMActorCriticPolicy(ActorCriticPolicy):

    def __init__(self,
                 observation_space,
                 action_space,
                 lr_schedule,
                 net_arch,
                 activation_fn=nn.Tanh,
                 *args,
                 **kwargs):
        super().__init__(observation_space,
                         action_space,
                         lr_schedule,
                         net_arch,
                         activation_fn,
                         features_extractor_class=LSTMFeatureExtractor,
                         *args,
                         **kwargs)

        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = LSTMModel(self.features_dim, **self.net_arch)


class LSTMTD3Policy(TD3Policy):

    def __init__(self,
                 observation_space,
                 action_space,
                 lr_schedule,
                 net_arch,
                 activation_fn=nn.Tanh,
                 *args,
                 **kwargs):
        super().__init__(observation_space,
                         action_space,
                         lr_schedule,
                         net_arch,
                         activation_fn,
                         features_extractor_class=LSTMFeatureExtractor,
                         *args,
                         **kwargs)
        self.ortho_inig = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = LSTMModel(self.features_dim, **self.net_arch)
