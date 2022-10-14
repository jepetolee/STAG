import torch
import torch.nn as nn

from BasicModel.Actor import ActorModel
from BasicModel.DataStructure import RSSMState
from BasicModel.DenseModel import DenseModel
from BasicModel.EncoderNDecoder import ObservationDecoder, ObservationEncoder
from BasicModel.RSSM import TransitionModel, RepresentationModel, RolloutModel


class TradingModel(nn.Module):
    def __init__(self, output_size, stochastic_size=30, deterministic_size=200,
                 hidden_size=200, action_hidden_size=200,
                 action_layers=3, action_dist='one_hot', reward_shape=(1,),
                 reward_hidden=300, value_shape=(1,), value_hidden=200):
        super().__init__()
        self.observation_encoder = ObservationEncoder()
        encoder_embed_size = self.observation_encoder.embed_size()
        embedding_size = 10000
        #need to check embedding size
        self.observation_decoder = ObservationDecoder(stochastic_size, deterministic_size, embedding_size)

        self.transition = TransitionModel(output_size, stochastic_size, deterministic_size, hidden_size)
        self.representation = RepresentationModel(encoder_embed_size, output_size, stochastic_size,
                                                  deterministic_size, hidden_size)
        self.rollout = RolloutModel(encoder_embed_size, output_size, stochastic_size,
                                    deterministic_size, hidden_size)
        feature_size = stochastic_size + deterministic_size
        self.action_size = output_size
        self.action_dist = action_dist
        self.action_decoder = ActorModel(output_size, feature_size, action_hidden_size, action_layers, action_dist)
        self.reward_model = DenseModel(feature_size, reward_shape, reward_hidden)
        self.value_model = DenseModel(feature_size, value_shape, value_hidden)
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size

    def forward(self, observation: torch.Tensor, prev_action: torch.Tensor = None, prev_state: RSSMState = None):
        state = self.get_state_representation(observation, prev_action, prev_state)
        action, action_distribution = self.policy(state)

        value = self.value_model(state.get_feature())
        reward = self.reward_model(state.get_feature())
        return action, action_distribution, value, reward, state

    def policy(self, state: RSSMState):
        feature = state.get_feature()
        action_dist = self.action_decoder(feature)
        if self.action_dist == 'tanh_normal':
            if self.training:
                action = action_dist.rsample()
            else:
                action = action_dist.mode()
        elif self.action_dist == 'one_hot':
            action = action_dist.sample()
            # This doesn't change the value, but gives us straight-through gradients
            action = action + action_dist.probs - action_dist.probs.detach()
        elif self.action_dist == 'relaxed_one_hot':
            action = action_dist.rsample()
        else:
            action = action_dist.sample()
        return action, action_dist

    def get_state_representation(self, observation: torch.Tensor, prev_action: torch.Tensor = None,
                                 prev_state: RSSMState = None):
        obs_embed = self.observation_encoder(observation)
        if prev_action is None:
            prev_action = torch.zeros(observation.size(0), self.action_size,
                                      device=observation.device, dtype=observation.dtype)
        if prev_state is None:
            prev_state = self.representation.initial_state(prev_action.size(0), device=prev_action.device,
                                                           dtype=prev_action.dtype)
        _, state = self.representation(obs_embed, prev_action, prev_state)
        return state

    def get_state_transition(self, prev_action: torch.Tensor, prev_state: RSSMState):
        state = self.transition(prev_action, prev_state)
        return state
