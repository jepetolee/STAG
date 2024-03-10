from Model import *
from DatasetBuilder import *
from utils import *
import pyarrow
#현물 호가, 선물호가 같이 보기 데이터로 삼기
data_columns = [
    'ClosePrice5', 'HighPrice5', 'LowPrice5', 'OpenPrice5', 'TwentyAvg5', 'FiftyAvg5', 'HundredAvg5', 'Volume5',
    'BollingerBandUpper5', 'BollingerBandLower5', 'RSI5', 'PercentK5', 'PercentD5', 'PercentJ5',
    'ClosePrice15', 'HighPrice15', 'LowPrice15', 'OpenPrice15', 'TwentyAvg15', 'FiftyAvg15', 'HundredAvg15', 'Volume15',
    'BollingerBandUpper15',
    'BollingerBandLower15', 'RSI15', 'PercentK15', 'PercentD15', 'PercentJ15',
    'ClosePrice1h', 'HighPrice1h', 'LowPrice1h', 'OpenPrice1h', 'TwentyAvg1h', 'FiftyAvg1h', 'HundredAvg1h', 'Volume1h',
    'BollingerBandUpper1h',
    'BollingerBandLower1h', 'RSI1h', 'PercentK1h', 'PercentD1h', 'PercentJ1h',
    'ClosePrice4h', 'HighPrice4h', 'LowPrice4h', 'OpenPrice4h', 'TwentyAvg4h', 'FiftyAvg4h', 'HundredAvg4h', 'Volume4h',
    'BollingerBandUpper4h',
    'BollingerBandLower4h', 'RSI4h', 'PercentK4h', 'PercentD4h', 'PercentJ4h',
    'ClosePrice1d', 'HighPrice1d', 'LowPrice1d', 'OpenPrice1d', 'TwentyAvg1d', 'FiftyAvg1d', 'HundredAvg1d', 'Volume1d',
    'BollingerBandUpper1d',
    'BollingerBandLower1d', 'RSI1d', 'PercentK1d', 'PercentD1d', 'PercentJ1d']


def TradeModel(crypto_name, leverage=20):
    TradeNet = TradingModel().float().cuda()
    TradeNet.eval()
    TradeNet.BuildMemory()
    TradeNet.load_state_dict(torch.load('./BaseModel/BaseModel' + crypto_name + '3.pt'))
    tensor_list = list()
    RealTrader = RL_Agent(leverage=leverage, testmode=True)
    testing_number_data = TakeCsvData('G:/CsvStorage/' + crypto_name + '/' + crypto_name + '_5M.csv')

    while 1:
        try:
            decided_number = 0  # numb + adder + mediate

            url = 'G:/ParquetDataStorage/' + crypto_name + '/' + str(decided_number + 1) + '.parquet'

            crypto_chart = pd.read_parquet(url, columns=data_columns)

            crypto_chart = crypto_chart.to_numpy()
            tensor_list.append(torch.from_numpy(crypto_chart).float())

            close_price_in_csv_data = testing_number_data['ClosePrice'][
                decided_number + 57399 + LeastNumber2Build]

            RealTrader.checking_bankrupt()
            RealTrader.check_price_type(close_price_in_csv_data)

            Profit, DoesDone = RealTrader.get_reward()

            if DoesDone:
                return
        except pyarrow.lib.ArrowInvalid:
            print(decided_number + 1)
            pass
        except KeyboardInterrupt:
            print("Finishing Trading With Keyboard")
            return
        except:
            print("UNKNOWN ERROR CAUSED")
            return


if __name__ == '__main__':
    model = TradingModel()
