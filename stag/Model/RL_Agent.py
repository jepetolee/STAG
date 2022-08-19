from stag.TradeManager import *

BANKRUPT_SCORE = -100
BENEFIT_SCORE = 10
LOSS_SCORE = -15
NONE_SCORE = 0

# POSITION_TYPES
POSITION_LONG = 'BUY'
POSITION_SHORT = 'SELL'
POSITION_HOLD = 'NONE'

# LEVERAGE_TYPES
LEVERAGE_HIGH = 5
LEVERAGE_DEFAULT = 3
LEVERAGE_LOW = 1

# CRYPTOTYPES
BTC_DECIMAL_POINT = 1000
ETH_DECIMAL_POINT = 1

THERES_NO_CRYPTO = 'NONE'

import numpy as np


class RL_Agent:
    def __init__(self, leverage, testmode=true):
        self.Agent = FutureTrader()
        self.RealTrader = self.Agent.Trader
        self.IsTestMode = testmode
        self.Leverage = leverage

        self.PercentFromOriginal = 100
        self.TradeCounts = 0
        self.WinCounts = 0

        self.CurrentReward = NONE_SCORE
        self.PositionPrice = 0
        self.CurrentPrice = 0
        self.CurrentPosition = POSITION_HOLD
        self.CurrentCallingSize = 0
        self.CurrentCryptoName = THERES_NO_CRYPTO

    def Trade(self, crypto_name, position, crypto_decimal_points):
        self.CurrentCryptoName = crypto_name
        self.PositionPrice = FutureTrader.CurrentPrice(self.CurrentCryptoName)
        self.CurrentPosition = position
        self.TradeCounts += 1

        if not self.IsTestMode:
            self.CurrentCallingSize = count_token_size(self.Leverage, self.Agent.CallableUsdt(), crypto_decimal_points)
            self.RealTrader.futures_create_order(symbol=self.CurrentCryptoName, type='LIMIT', timeInForce='GTC',
                                                 price=self.PositionPrice, side=self.CurrentPosition
                                                 , quantity=self.CurrentCallingSize)
        return

    def FinishTrade(self):
        self.CurrentPrice = FutureTrader.CurrentPrice(self.CurrentCryptoName)
        if not self.IsTestMode:
            self.CurrentCallingSize = count_token_size(self.Leverage, self.Agent.CallableUsdt(), crypto_decimal_points)
            self.RealTrader.futures_create_order(symbol=self.CurrentCryptoName, type='LIMIT', timeInForce='GTC',
                                                 price=self.CurrentPrice, side=self.CurrentPosition
                                                 , quantity=self.CurrentCallingSize)

        self.CurrentCryptoName = THERES_NO_CRYPTO
        self.CurrentPosition = POSITION_HOLD
        if self.CurrentReward > 0:
            self.WinCounts += 1

        # need to count percent and set closing position


def count_token_size(leverage, usdt_size, crypto_decimal_points):
    leveraged_usdt = leverage * usdt_size
    floored_size = np.floor(crypto_decimal_points * leveraged_usdt / self.PositionPrice)
    calling_size = int(floored_size / crypto_decimal_points)

    return float(calling_size)
