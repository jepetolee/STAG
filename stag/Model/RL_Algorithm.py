import torch

from RL_Model import *


class Dreamer:
    def __init__(self, device):
        self.trading_model = TradingModel(output_size=3)
        self.device = device

    def hypothesis(self, crypto_chart):
        Model = self.trading_model
        reward,observation = list(),list()  # 이거 원리 확인 해야함
        embedded_data = Model.observation_encoder(crypto_chart)
        previous_state = Model.representation.initial_state(device=self.device, dtype=torch.float64)
        prior_ones, posterior_ones = Model.rollout.Representation(embedded_data, action, previous_state)

        feature = posterior_ones.get_feature()
        image_prediction = Model.observation_decoder(feature)
        reward_prediction = Model.reward_model(feature)
        return image_prediction