from stag.TradeManager import *


def test():
    trader = FutureTrader()
    try:
        trader.CurrentPrice('BTCUSDT')
    finally:
        print("Checking current price doesn't Work.")

    try:
        trader.CallableUsdt()
    finally:
        print("Account calling doesn't work.")