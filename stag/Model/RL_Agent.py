from .TradeManager import *
import numpy as np
import torch

BANKRUPT_CONSTANT = -100

# POSITION_TYPES
POSITION_HOLD = 2
POSITION_LONG = 1
POSITION_SHORT = 0

# CRYPTOTYPES
BTC_DECIMAL_POINT = 1000
ETH_DECIMAL_POINT = 1

THERES_NO_CRYPTO = 'NONE'

CheckActionSelectModeTrue = "TRUE"
CheckActionSelectModeUNTRAINED = "UNTRAINED"
CheckActionSelectModeFalse = "False"


class RL_Agent:
    def __init__(self, leverage, testmode=True):
        if not testmode:
          self.Agent = FutureTrader()
          self.RealTrader = self.Agent.Trader
        self.IsTestMode = testmode
        self.Leverage = leverage
        self.PercentState = 100
        self.UndefinedPercent = 100
        self.CheckActionSelectMode = CheckActionSelectModeTrue
        self.ChoicingTensor = torch.zeros(1, 1)
        self.PositionPrice = 0
        self.CurrentPrice = 0

        self.CurrentPosition = POSITION_HOLD
        self.CurrentCallingSize = 0
        self.CurrentCryptoName = THERES_NO_CRYPTO

    def Trade(self, crypto_name, crypto_decimal_points):
        self.CurrentCryptoName = crypto_name

        self.PositionPrice = FutureTrader.CurrentPrice(self.CurrentCryptoName)
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
            self.UndefinedPercent = 100
            return True
    def conversion_constant(self):
        if self.CurrentPosition is POSITION_LONG:
            revenue = 0.9998 * self.CurrentPrice - self.PositionPrice
            profit_percent = (revenue / self.PositionPrice) * 100
            conversion_constant = profit_percent * self.Leverage

            Leveraged_change = (conversion_constant + 100) / 100
            self.UndefinedPercent = self.PercentState * Leveraged_change

        elif self.CurrentPosition is POSITION_SHORT:
            revenue = 0.9998 * self.PositionPrice - self.CurrentPrice
            profit_percent = (revenue / self.PositionPrice) * 100
            conversion_constant = profit_percent * self.Leverage

            Leveraged_change = (conversion_constant + 100) / 100
            self.UndefinedPercent = self.PercentState * Leveraged_change

        elif self.CurrentPosition is POSITION_HOLD:
            brevenue = 0.9998 * self.CurrentPrice - self.PositionPrice
            srevenue = 0.9998 * self.PositionPrice - self.CurrentPrice
            revenue = min(brevenue, srevenue)
            profit_percent = (revenue / self.PositionPrice) * 100
            conversion_constant = profit_percent * self.Leverage
            Leveraged_change = 1
        return conversion_constant,Leveraged_change
    def get_reward(self):

        conversion_constant, Leveraged_change = self.conversion_constant()
        DoesDone = self.checking_bankrupt()

        if self.CheckActionSelectMode == CheckActionSelectModeUNTRAINED:
            self.PercentState *= Leveraged_change
            if self.CurrentPosition is POSITION_HOLD:
                self.ValueSave(conversion_constant / 100)
            elif conversion_constant >= 0:
                self.ValueSave(conversion_constant / 100)
            elif conversion_constant < 0:
                self.ValueSave(conversion_constant / 100)
            reward = torch.tensor([[conversion_constant / 100]])
        else:
            reward = torch.tensor([[conversion_constant / 2000]])
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

                self.ProfitTracker = False
                self.TargetProfit = 200

        elif action.item() == 1:
            self.CheckActionSelectMode = CheckActionSelectModeFalse

    def check_position(self, action):

        self.CheckActionSelectMode = CheckActionSelectModeFalse
        if action.item() == 0:
            self.CurrentPosition =POSITION_LONG
        elif action.item() == 1:
            self.CurrentPosition = POSITION_SHORT
        elif action.item() == 2:
            self.CurrentPosition = POSITION_HOLD
        else:
            print("there is an error for deciding position")
        return self.CurrentPosition