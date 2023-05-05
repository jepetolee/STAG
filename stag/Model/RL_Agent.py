from stag.TradeManager import *
import numpy as np
import torch

BANKRUPT_CONSTANT = -100

# POSITION_TYPES
POSITION_HOLD = 0
POSITION_LONG = 1
POSITION_SHORT = 2


# LEVERAGE_TYPES
LEVERAGE_HIGH = 5
LEVERAGE_DEFAULT = 3
LEVERAGE_LOW = 1

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
        self.RealTrader = self.Agent.Trader
        self.IsTestMode = testmode
        self.Leverage = leverage
        self.HoldingCount =0
        self.PercentState = 100
        self.UndefinedPercent = 100
        self.TradeCounts = 0
        self.CheckActionSelectMode = True

        self.ChoicingTensor =  torch.zeros(1,3)

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
            self.CheckActionSelectMode = True
            self.PercentState = 100
            self.ChoicingTensor = torch.zeros(1, 3)
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

        self.UndefinedPercent = 100 * Leveraged_change

        if self.CurrentPosition is POSITION_HOLD:
            conversion_constant = 0
            self.HoldingCount +=1

        DoesDone = self.checking_bankrupt()

        if DoesDone is True:
            reward = BANKRUPT_CONSTANT
            print("BANKRUPTED")
        elif self.CheckActionSelectMode:
            self.HoldingCount=0
            self.ChoicingTensor = torch.zeros(1, 3)
            self.PercentState *= Leveraged_change
            reward = conversion_constant
        else:
            if conversion_constant ==0:
                reward = -self.HoldingCount/13
            else:
                reward = conversion_constant/10

        if reward >=0:
            self.ChoicingTensor[0][self.CurrentPosition]+=reward
            self.ChoicingTensor[0][0]=0
        else:
            self.ChoicingTensor -= reward
            self.ChoicingTensor[0][self.CurrentPosition] += reward
            self.ChoicingTensor[0][0] = 0

        return reward

    def check_price_type(self, price):
        if self.CheckActionSelectMode:
            self.PositionPrice = price
            self.CheckActionSelectMode =False
        else:
            pass
        self.CurrentPrice = price

    def check_keeping(self,action):
        if action.item() == 0:
            self.CheckActionSelectMode = False
        elif action.item() == 1:
            self.CheckActionSelectMode = True

    def check_position(self, action):
        self.TradeCounts += 1
        if action.item() == 0:
            self.CurrentPosition = POSITION_HOLD
        elif action.item() == 1:
            self.CurrentPosition = POSITION_LONG
        elif action.item() == 2:
            self.CurrentPosition = POSITION_SHORT
        else:
            print("there is an error for deciding position")
        return self.CurrentPosition
