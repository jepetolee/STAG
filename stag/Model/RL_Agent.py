from stag.TradeManager import *
import numpy as np

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

# ACTION_CHANGED
UNSTARTED = 3
POSITION_UNCHANGED = 4
POSITION_CHANGED = 5

# CRYPTOTYPES
BTC_DECIMAL_POINT = 1000
ETH_DECIMAL_POINT = 1

THERES_NO_CRYPTO = 'NONE'


def ChangePosition2Integer(position):
    if position is POSITION_HOLD:
        return 0
    elif position is POSITION_LONG:
        return 1
    elif position is POSITION_SHORT:
        return 2
    else:
        print("It has error to decide Position")


class RL_Agent:
    def __init__(self, leverage, testmode=true):
        self.Agent = FutureTrader()
        self.CheckActionChanged = UNSTARTED
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

    def Trade(self, crypto_name, crypto_decimal_points):
        self.CurrentCryptoName = crypto_name

        self.PositionPrice = FutureTrader.CurrentPrice(self.CurrentCryptoName)
        self.TradeCounts += 1

        if not self.IsTestMode:
            self.CurrentCallingSize = self.count_token_size(self.Leverage, self.Agent.CallableUsdt(),
                                                            crypto_decimal_points)
            self.RealTrader.futures_create_order(symbol=self.CurrentCryptoName, type='LIMIT', timeInForce='GTC',
                                                 price=self.PositionPrice, side=self.CurrentPosition,
                                                 quantity=self.CurrentCallingSize)
        return

    def FinishTrade(self, crypto_decimal_points):
        self.CurrentPrice = FutureTrader.CurrentPrice(self.CurrentCryptoName, self.CurrentCryptoName)
        if not self.IsTestMode:
            self.CurrentCallingSize = self.count_token_size(self.Leverage, self.Agent.CallableUsdt(),
                                                            crypto_decimal_points)
            self.RealTrader.futures_create_order(symbol=self.CurrentCryptoName, type='LIMIT', timeInForce='GTC',
                                                 price=self.CurrentPrice, side=self.CurrentPosition,
                                                 quantity=self.CurrentCallingSize)

        self.CurrentCryptoName = THERES_NO_CRYPTO
        self.CurrentPosition = POSITION_HOLD
        if self.CurrentReward > 0:
            self.WinCounts += 1

        # need to count percent and set closing position

    def count_token_size(self, leverage, usdt_size, crypto_decimal_points):
        leveraged_usdt = leverage * usdt_size
        floored_size = np.floor(crypto_decimal_points * leveraged_usdt / self.PositionPrice)
        calling_size = int(floored_size / crypto_decimal_points)

        return float(calling_size)

    def get_reward(self):
        return

    def check_price_type(self, price):
        if self.CheckActionChanged is POSITION_CHANGED:
            self.PositionPrice = price
            self.CheckActionChanged = POSITION_UNCHANGED
        else: pass
        self.CurrentPrice = price

    def check_position(self, action):
        if self.CheckActionChanged is UNSTARTED:
            self.CheckActionChanged = POSITION_CHANGED
            return self.decide_position(action)
        elif ChangePosition2Integer(self.CurrentPosition) == action.item():
            self.CheckActionChanged = POSITION_UNCHANGED
            return self.CurrentPosition
        else:
            self.CheckActionChanged = POSITION_CHANGED
            return self.decide_position(action)

    def decide_position(self, action):
        if action.item() == 0:
            self.CurrentPosition = POSITION_HOLD
        elif action.item() == 1:
            self.CurrentPosition = POSITION_LONG
        elif action.item() == 2:
            self.CurrentPosition = POSITION_SHORT
        else:
            print("there is an error for deciding position")
        return self.CurrentPosition
