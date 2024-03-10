import sys
import pandas as pd

from tqdm import trange
from ModifyCsv import *
from TechnicalIndicator import *

sys.path.append('..')
Crypto_LINK = []
position_list = []
profit_list = []
LeastNumber2Build = 240  # 0~199
IMGFileMainRoot = 'G:/ImgDataStorage/'
ParquetMainRoot = 'G:/NumpyDataStorage/'

def BuildCryptoData(crypto_name, adder=0, endsize=30000,levrage=20,futureValue=288,target_profit=1):

    adder1, adder5, adder15, adder1h, adder4h, adder1d = 287799, 57399, 18999, 4599, 999, 0

    url5, url15, url1h, url4h, url1d = r'G:/CsvStorage/' + crypto_name + '/' + crypto_name + '_5M.csv', \
                                             r'G:/CsvStorage/' + crypto_name + '/' + crypto_name + '_15M.csv', \
                                             r'G:/CsvStorage/' + crypto_name + '/' + crypto_name + '_1H.csv', \
                                             r'G:/CsvStorage/' + crypto_name + '/' + crypto_name + '_4H.csv', \
                                             r'G:/CsvStorage/' + crypto_name + '/' + crypto_name + '_1D.csv'

    PandasArray5m = TakeCsvData(url5)

    data_size = PandasArray5m.shape[0]

    NumbersOfDataset = int(data_size - LeastNumber2Build - adder5 - adder)
    if endsize ==0:
        NumbersOfDataset = int(data_size - LeastNumber2Build - adder5 - adder)
    elif NumbersOfDataset<endsize :
        NumbersOfDataset = int(data_size - LeastNumber2Build - adder5 - adder)
    else:
        NumbersOfDataset=endsize

    for temp in trange(NumbersOfDataset):

        temp = temp + adder
        starting_x1 = temp + adder5



        processed_data5 = PandasArray5m.iloc[starting_x1]['ClosePrice']
        future_data5 = PandasArray5m.iloc[starting_x1 + 1 + futureValue]['ClosePrice']

        parquet_link = ParquetMainRoot + crypto_name + '/' + str(temp + 1) + '.npy'

        if levrage * (future_data5 * 0.9998 - processed_data5) / processed_data5 * 100 >= target_profit:
            Crypto_LINK.append(parquet_link)
            position_list.append(0)


        elif levrage * (processed_data5 * 0.9998 - future_data5) / processed_data5 * 100 * (-1) >= target_profit:
            Crypto_LINK.append(parquet_link)
            position_list.append(1)

        else:
            Crypto_LINK.append(parquet_link)
            position_list.append(2)
        profit_list.append(levrage * (future_data5 * 0.9998 - processed_data5) / processed_data5 * 100)


if __name__ == '__main__':
    BuildCryptoData('LTCUSDT',30000, 330000,1)

    data = {'url': Crypto_LINK, 'POSITION': position_list,"profit": profit_list}
    df = pd.DataFrame(data)
    df.to_csv('./LTC_FUTURUE150_20_PROFIT.csv', index=True)
    Crypto_LINK.clear()
    position_list.clear()
    profit_list.clear()


    BuildCryptoData('BCHUSDT',30000, 330000,1)

    data = {'url': Crypto_LINK, 'POSITION': position_list,"profit": profit_list}
    df = pd.DataFrame(data)
    df.to_csv('./BCH_FUTURUE150_20_PROFIT.csv', index=True)
    Crypto_LINK.clear()
    position_list.clear()
    profit_list.clear()


    BuildCryptoData('BTCUSDT', 30000, 360000, 1)

    data = {'url': Crypto_LINK, 'POSITION': position_list, "profit": profit_list}
    df = pd.DataFrame(data)
    df.to_csv('./BTC_FUTURUE150_20_PROFIT.csv', index=True)
    Crypto_LINK.clear()
    position_list.clear()
    profit_list.clear()

    BuildCryptoData('ETHUSDT', 30000, 350000, 1)

    data = {'url': Crypto_LINK, 'POSITION': position_list, "profit": profit_list}
    df = pd.DataFrame(data)
    df.to_csv('./ETH_FUTURUE150_20_PROFIT.csv', index=True)
    Crypto_LINK.clear()
    position_list.clear()
    profit_list.clear()

    BuildCryptoData('SOLUSDT', 30000, 270000, 1)

    data = {'url': Crypto_LINK, 'POSITION': position_list, "profit": profit_list}
    df = pd.DataFrame(data)
    df.to_csv('./SOL_FUTURUE150_20_PROFIT.csv', index=True)
    Crypto_LINK.clear()
    position_list.clear()
    profit_list.clear()

    BuildCryptoData('XRPUSDT', 30000, 330000, 1)

    data = {'url': Crypto_LINK, 'POSITION': position_list, "profit": profit_list}
    df = pd.DataFrame(data)
    df.to_csv('./XRP_FUTURUE150_20_PROFIT.csv', index=True)
    Crypto_LINK.clear()
    position_list.clear()
    profit_list.clear()