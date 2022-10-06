import torch

from RL_Model import *


class Dreamer:
    def __init__(self, device):
        self.trading_model = TradingModel(output_size=3)
        self.device = device

    def hypothesis(self, crypto_chart):
        return self.trading_model(crypto_chart)

