import numpy as np
import torch
import torch.nn as nn

from BasicModel.Actor import ActorModel
from BasicModel.EncoderNDecoder import ObservationDecoder, ObservationEncoder
from BasicModel.RSSM import TransitionModel, RepresentationModel, RolloutModel
from BasicModel.DenseModel import DenseModel

class TradingModel(nn.Module):
    def __init__(self, action_shape, stochastic_size=30,deterministic_size=200,
                 hidden_size=200, image_shape=(3, 500, 500), action_hidden_size=200,
                 action_layers=3, action_dist='one_hot', reward_shape=(1,),
                 reward_hidden=300, value_shape=(1,), value_hidden=200):
        super().__init__()
        self.observation_encoder = ObservationEncoder()
        encoder_embed_size = self.observation_encoder.embed_size
        decoder_embed_size = stochastic_size + deterministic_size
        self.observation_decoder = ObservationDecoder(embedding_size=decoder_embed_size)
        self.action_shape = action_shape
        output_size = np.prod(action_shape)
        self.transition = TransitionModel(output_size, stochastic_size, deterministic_size, hidden_size)
        self.representation = RepresentationModel(encoder_embed_size, output_size, stochastic_size,
                                                 deterministic_size, hidden_size)
        self.rollout = RolloutModel(self.representation, self.transition)
        feature_size = stochastic_size + deterministic_size
        self.action_size = output_size
        self.action_dist = action_dist
        self.action_decoder = ActorModel(output_size, feature_size, action_hidden_size, action_layers, action_dist)
        self.reward_model = DenseModel(feature_size, reward_shape, reward_hidden)
        self.value_model = DenseModel(feature_size, value_shape,  value_hidden)
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size

    def forward(self, observation: torch.Tensor, prev_action: torch.Tensor = None, prev_state: RSSMState = None):
        state = self.get_state_representation(observation, prev_action, prev_state)
        action, action_dist = self.policy(state)
        value = self.value_model(get_feat(state))
        reward = self.reward_model(get_feat(state))
        return action, action_dist, value, reward, state

    def policy(self, state: RSSMState):
        feat = get_feat(state)
        action_dist = self.action_decoder(feat)
        if self.action_dist == 'tanh_normal':
            if self.training:  # use agent.train(bool) or agent.eval()
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

    def forward(self, observation: torch.Tensor, prev_action: torch.Tensor = None, prev_state: RSSMState = None):
        return_spec = ModelReturnSpec(None, None)
        raise NotImplementedError()
