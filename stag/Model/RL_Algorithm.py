from RL_Model import *


class Dreamer:
    def __init__(self):
        self.trading_model = TradingModel(output_size=3)

    def hypothesis(self, crypto_chart):
        self.trading_model(crypto_chart)
