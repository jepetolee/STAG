from stag.TradeManager import *
import numpy as np
import torch

BANKRUPT_CONSTANT = -100

# POSITION_TYPES
POSITION_HOLD = 0
POSITION_LONG = 1
POSITION_SHORT = 2

# CRYPTOTYPES
BTC_DECIMAL_POINT = 1000
ETH_DECIMAL_POINT = 1

THERES_NO_CRYPTO = 'NONE'

CheckActionSelectModeTrue = "TRUE"
CheckActionSelectModeUNTRAINED = "UNTRAINED"
CheckActionSelectModeFalse = "False"


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
    def __init__(self, leverage, testmode=True, limitymode=False, limit=-30):
        self.Agent = FutureTrader()
        self.RealTrader = self.Agent.Trader
        self.IsTestMode = testmode
        self.Leverage = leverage
        self.HoldingCount = 0
        self.PercentState = 100
        self.UndefinedPercent = 100
        self.TradeCounts = 0
        self.CheckActionSelectMode = CheckActionSelectModeTrue
        self.ChoicingTensor = torch.zeros(1, 1)
        self.PositionPrice = 0
        self.CurrentPrice = 0
        self.CurrentPosition = POSITION_HOLD
        self.CurrentCallingSize = 0
        self.CurrentCryptoName = THERES_NO_CRYPTO
        if limitymode:
            self.limity_mode = True
            self.limit = limit
        else:
            self.limity_mode = False

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

        return

        # need to count percent and set closing position

    def count_token_size(self, leverage, usdt_size, crypto_decimal_points):
        leveraged_usdt = leverage * usdt_size
        floored_size = np.floor(crypto_decimal_points * leveraged_usdt / self.PositionPrice)
        calling_size = int(floored_size / crypto_decimal_points)

        return float(calling_size)

    def checking_bankrupt(self):
        if self.UndefinedPercent > 5:
            return False
        else:
            if self.IsTestMode:

                self.CheckActionSelectMode = CheckActionSelectModeUNTRAINED
            else:
                self.CheckActionSelectMode = CheckActionSelectModeTrue
            self.PercentState = 100
            return True

    def get_reward(self):
        revenue = 0
        if self.CurrentPosition is POSITION_LONG:
            revenue = 0.9998 * self.CurrentPrice - self.PositionPrice
        elif self.CurrentPosition is POSITION_SHORT:
            revenue = 0.9998 * self.PositionPrice - self.CurrentPrice
        elif self.CurrentPosition is POSITION_HOLD:
            revenue = 0  # least loss that prevent non-decision phenomenon

        profit_percent = (revenue / self.PositionPrice) * 100
        conversion_constant = profit_percent * self.Leverage

        Leveraged_change = (conversion_constant + 100) / 100
        self.UndefinedPercent = self.PercentState * Leveraged_change

        DoesDone = self.checking_bankrupt()

        if self.limity_mode is True and conversion_constant < self.limit:
            self.PercentState *= Leveraged_change
            reward = conversion_constant / 100
            self.CheckActionSelectMode = CheckActionSelectModeUNTRAINED
            self.ValueSave(-0.6)

        elif self.CheckActionSelectMode == CheckActionSelectModeUNTRAINED:

            self.PercentState *= Leveraged_change
            reward = conversion_constant / 100
            self.ValueSave(conversion_constant*2)

        else:
            if self.CurrentPosition is POSITION_HOLD:
                reward = 0
                self.ValueSave(reward)
            else:
                reward =conversion_constant/100

        return reward, DoesDone

    def check_price_type(self, price):
        self.CurrentPrice = price

    def ValueSave(self, reward):
        self.ChoicingTensor[0] = reward
        return

    def check_keeping(self, action):
        if action.item() == 0:
            if self.IsTestMode:
                self.CheckActionSelectMode = CheckActionSelectModeUNTRAINED
            else:
                self.CheckActionSelectMode = CheckActionSelectModeTrue
        elif action.item() == 1:
            self.CheckActionSelectMode = CheckActionSelectModeFalse

    def check_position(self, action):

        self.CheckActionSelectMode = CheckActionSelectModeFalse
        self.TradeCounts += 1
        if action.item() == 0:
            self.CurrentPosition = POSITION_HOLD
            self.HoldingCount += 1
        elif action.item() == 1:
            self.CurrentPosition = POSITION_LONG
            self.HoldingCount = 0
        elif action.item() == 2:
            self.CurrentPosition = POSITION_SHORT
            self.HoldingCount = 0
        else:
            print("there is an error for deciding position")
        return self.CurrentPosition
