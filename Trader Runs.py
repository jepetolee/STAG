from predict import *
from stag.Model.TradeManager import *
import time
device = 'cpu'
System = FutureTrader()
vote = [0,0,0]
BestPredictScore = 0
ChoosedOne ='None'
ChoosedPosition = 'None'
Dataset = ['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'XRPUSDT', 'BCHUSDT', 'SOLUSDT','EOSUSDT','ETCUSDT']
Leverage = {'BTCUSDT':20,
            'ETHUSDT':15,
            'LTCUSDT':10,
            'XRPUSDT':10,
            'BCHUSDT':10,
            'SOLUSDT':10,
            'EOSUSDT':10
            }



def count_token_size(ChoosedOne, leverage, usdt_size,PositionPrice):
    leveraged_usdt = leverage * usdt_size/PositionPrice
    url = f'https://api.binance.com/api/v3/exchangeInfo?symbol={ChoosedOne}'

    response = requests.get(url)
    data = response.json()

    # 최소 주문량 정보 확인
    min_quantity = float(data['symbols'][0]['filters'][1]['minQty'])

    step_size = float(data['symbols'][0]['filters'][2]['stepSize'])
    # quantity를 사이즈 단위에 맞추기
    quantity = round(leveraged_usdt / step_size) * step_size

    if quantity < min_quantity:
        print(f"주문량이 최소 주문량 미만입니다. 최소 주문량은 {min_quantity} {ChoosedOne}입니다.")
        return

    return quantity



def FinishTrade(self, crypto_decimal_points):
    self.CurrentPrice = FutureTrader.CurrentPrice(self.CurrentCryptoName, self.CurrentCryptoName)
    if not self.IsTestMode:
        self.CurrentCallingSize = self.count_token_size(self.Leverage, self.Agent.CallableUsdt(),
                                                        crypto_decimal_points)
        self.RealTrader.futures_create_order(symbol=self.CurrentCryptoName, type='LIMIT', timeInForce='GTC',
                                             price=self.CurrentPrice, side=self.CurrentPosition,
                                             quantity=self.CurrentCallingSize)

    return

while True:

    for key in Dataset:

        position, value =predict(key,device)
        print(position,value )
        vote[position] += 1
        if value > BestPredictScore:
            BestPredictScore = value
            ChoosedPosition= position
            ChoosedOne = key

    print(vote,BestPredictScore,System.CallableUsdt(),ChoosedOne)
    vote = [0,0,0]
    BestPredictScore = 0
    time.sleep(3600)




    '''  System.Trader.futures_coin_change_leverage(symbol=ChoosedOne,leverage=Leverage[ChoosedOne])
    server_time = System.Trader.get_server_time()['serverTime']
    CurrentCallingSize = count_token_size(ChoosedOne,Leverage[ChoosedOne],System.CallableUsdt())
    if ChoosedPosition ==0:
         System.Trader.futures_create_order(symbol=ChoosedOne, side='BUY', type='MARKET', positionSide='LONG',
                                            quantity=CurrentCallingSize, timestamp=server_time)
         System.CurrentPrice(ChoosedOne)
         System.Trader.futures_create_order(symbol=ChoosedOne, side='SELL', type='MARKET', positionSide='LONG',
                                            quantity=CurrentCallingSize, timestamp=server_time, closePosition=True)
    elif ChoosedPosition ==1:
         System.Trader.futures_create_order(symbol=ChoosedOne, side='SELL', type='MARKET', positionSide='SHORT',
                                            quantity=CurrentCallingSize, timestamp=server_time)
         System.Trader.futures_create_order(symbol=ChoosedOne, side='BUY', type='MARKET', positionSide='SHORT',
                                            quantity=CurrentCallingSize, timestamp=server_time, closePosition=True)
    else:
        CurrentPosition =  'HOLD'''




