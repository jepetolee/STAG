from stag.TradeManager import *
import numpy as np
import torch

BANKRUPT_CONSTANT = -1000
BENEFIT_CONSTANT = 1
LOSS_CONSTANT = 2

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
    def __init__(self, leverage, testmode=True):
        self.Agent = FutureTrader()
        self.CheckActionChanged = UNSTARTED
        self.RealTrader = self.Agent.Trader
        self.IsTestMode = testmode
        self.Leverage = leverage

        self.PercentState = 100
        self.UndefinedPercent = 100
        self.TradeCounts = 0
        self.WinCounts = 0
        self.SavedBenefit = 0

        self.CurrentReward = 0
        self.PositionPrice = 0
        self.CurrentPrice = 0
        self.FirstTicket = True
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
        return

        # need to count percent and set closing position

    def count_token_size(self, leverage, usdt_size, crypto_decimal_points):
        leveraged_usdt = leverage * usdt_size
        floored_size = np.floor(crypto_decimal_points * leveraged_usdt / self.PositionPrice)
        calling_size = int(floored_size / crypto_decimal_points)

        return float(calling_size)

    def checking_state(self):
        if self.PercentState > 5:
            return False
        else:
            return True

    def get_reward(self):
        revenue = 0
        if self.CurrentPosition is POSITION_LONG:
            revenue = 0.9998 * self.CurrentPrice - self.PositionPrice
        elif self.CurrentPosition is POSITION_SHORT:
            revenue = 0.9998 * self.PositionPrice - self.CurrentPrice
        elif self.CurrentPosition is POSITION_HOLD:
            revenue = -0.1  # least loss that prevent non-decision phenomenon

        profit_percent = (revenue / self.PositionPrice) * 100
        conversion_constant = profit_percent * self.Leverage
        Leveraged_change = (conversion_constant + 100)/100
        self.UndefinedPercent = self.PercentState * Leveraged_change

        if self.CurrentPosition is POSITION_HOLD:
            conversion_constant = -1
        else:
            temp = conversion_constant
            conversion_constant -= self.SavedBenefit
            self.SavedBenefit = temp

        DoesDone = self.checking_state()
        if DoesDone is True:
            reward = BANKRUPT_CONSTANT
            self.SavedBenefit = 0
            self.PercentState = 100
        elif self.CheckActionChanged is POSITION_CHANGED:
            print("changed")
            if self.FirstTicket is True:
                self.FirstTicket = False
            else:
                self.PercentState *= Leveraged_change
                self.CheckActionChanged = POSITION_UNCHANGED
                self.SavedBenefit = 0
            reward = conversion_constant
        else:
            if conversion_constant >= 0:
                reward = conversion_constant * BENEFIT_CONSTANT

            else:
                reward = conversion_constant * LOSS_CONSTANT

        return reward

    def check_price_type(self, price):
        if self.CheckActionChanged is POSITION_CHANGED:
            self.PositionPrice = price
        else:
            pass
        self.CurrentPrice = price

    def check_position(self, action):
        action = torch.argmax(action)
        if self.CheckActionChanged is UNSTARTED:
            self.CheckActionChanged = POSITION_CHANGED
            self.TradeCounts += 1
            return self.decide_position(action)
        elif ChangePosition2Integer(self.CurrentPosition) == action.item():
            self.CheckActionChanged = POSITION_UNCHANGED
            return self.CurrentPosition
        else:
            self.CheckActionChanged = POSITION_CHANGED
            self.TradeCounts += 1
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
