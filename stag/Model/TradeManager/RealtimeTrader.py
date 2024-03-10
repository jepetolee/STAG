from binance.client import Client
from binance.exceptions import BinanceAPIException

# ERRORCODE
NONE_AFFORDABLE = -1
PRICE_CHECKING_FAILED = -2


def GetApiKey():
    API = open('G:/STAG/stag/Model/TradeManager/ApiKeyStorage/Api', 'r')
    ApiKey, ApiSecret = API.readline().split(',')
    return ApiKey, ApiSecret


def GetTestApiKey():
    API = open('G:/STAG/stag/Model/TradeManager/ApiKeyStorage/TestApi', 'r')
    ApiKey, ApiSecret = API.readline().split(',')
    return ApiKey, ApiSecret


class FutureTrader:
    def __init__(self, UsingTest=False):

        if UsingTest:
            ApiKey, ApiSecret = GetTestApiKey()
        else:
            ApiKey, ApiSecret = GetApiKey()

        self.Trader = Client(api_key=ApiKey, api_secret=ApiSecret, testnet=UsingTest)
        self.Account = NONE_AFFORDABLE
        self.CallableUsdt()

    def CallableUsdt(self):
        try:
            AccountInformation = self.Trader.futures_account()

            for asset_type in AccountInformation["assets"]:
                if asset_type["asset"] == "USDT":
                    self.Account = float(asset_type["availableBalance"])
                    break

            if len(AccountInformation) > 0:
                self.Account = float("{:.2f}".format(self.Account))
            return self.Account

        except BinanceAPIException as e:
            print("Account connection was failed. Please check your Api keys")
            return NONE_AFFORDABLE

    def CurrentPrice(self, CryptoSymbol):
        try:
            return float(self.Trader.futures_symbol_ticker(symbol=CryptoSymbol)['price'])

        except BinanceAPIException as e:
            print("The crypto's current price was not updated.")
            return PRICE_CHECKING_FAILED
