import sys
import numpy as np
import pandas as pd


from stag.DatasetBuilder.ModifyCsv import *
from tqdm import trange
from stag.DatasetBuilder.TechnicalIndicator import *

sys.path.append('..')

LeastNumber2Build = 240  # 0~199
IMGFileMainRoot = 'G:/ImgDataStorage/'
ParquetMainRoot = 'G:/NumpyDataStorage/'


def BuildOneChartData(PandasArray: pd.DataFrame):


    OpenPrices = PandasArray['OpenPrice'].copy()
    HighPrices = PandasArray['HighPrice'].copy()
    LowPrices = PandasArray['LowPrice'].copy()
    ClosePrices = PandasArray['ClosePrice'].copy()

    OpenPrice = normalize_data(OpenPrices.values.reshape(1,-1))
    HighPrice = normalize_data(HighPrices.values.reshape(1,-1))
    LowPrice = normalize_data(LowPrices.values.reshape(1,-1))
    ClosePrice = normalize_data(ClosePrices.values.reshape(1,-1))
    Volume = normalize_data(PandasArray['Volume'].values.reshape(1,-1))

    MovingAverage5 = MovingAverageWithMaxAbsScaler(ClosePrices, 5)
    MovingAverage10 = MovingAverageWithMaxAbsScaler(ClosePrices, 10)
    MovingAverage40 = MovingAverageWithMaxAbsScaler(ClosePrices, 40)
    MovingAverage80 = MovingAverageWithMaxAbsScaler(ClosePrices, 80)

    BollingerBand12 = BollingerBandWithMaxAbsScaler(ClosePrices, 20, 2)
    BollingerBand20 = BollingerBandWithMaxAbsScaler(ClosePrices,20,2)

    RSI7 = RSIValue(ClosePrices,7)
    RSI14 = RSIValue(ClosePrices, 14)
    RSI25 = RSIValue(ClosePrices, 25)

    KDJ7= KDJValue(HighPrices,LowPrices,ClosePrices,7)
    KDJ14 = KDJValue(HighPrices, LowPrices, ClosePrices, 14)
    KDJ25= KDJValue(HighPrices, LowPrices, ClosePrices, 25)

    OpenPricesCurrent = PandasArray['OpenPriceCurrent'].copy()
    HighPricesCurrent = PandasArray['HighPriceCurrent'].copy()
    LowPricesCurrent = PandasArray['LowPriceCurrent'].copy()
    ClosePricesCurrent = PandasArray['ClosePriceCurrent'].copy()

    OpenPriceCurrent = normalize_data(OpenPricesCurrent.values.reshape(1,-1))
    HighPriceCurrent = normalize_data(HighPricesCurrent.values.reshape(1,-1))
    LowPriceCurrent = normalize_data(LowPricesCurrent.values.reshape(1,-1))
    ClosePriceCurrent = normalize_data(ClosePricesCurrent.values.reshape(1,-1))
    VolumeCurrent = normalize_data(PandasArray['VolumeCurrent'].values.reshape(1,-1))

    MovingAverage5Current = MovingAverageWithMaxAbsScaler(ClosePricesCurrent, 5)
    MovingAverage10Current = MovingAverageWithMaxAbsScaler(ClosePricesCurrent, 10)
    MovingAverage40Current = MovingAverageWithMaxAbsScaler(ClosePricesCurrent, 40)
    MovingAverage80Current = MovingAverageWithMaxAbsScaler(ClosePricesCurrent, 80)

    BollingerBand12Current = BollingerBandWithMaxAbsScaler(ClosePricesCurrent, 20, 2)
    BollingerBand20Current = BollingerBandWithMaxAbsScaler(ClosePricesCurrent,20,2)

    RSI7Current = RSIValue(ClosePricesCurrent,7)
    RSI14Current = RSIValue(ClosePricesCurrent, 14)
    RSI25Current = RSIValue(ClosePricesCurrent, 25)

    KDJ7Current = KDJValue(HighPricesCurrent,LowPricesCurrent,ClosePricesCurrent,7)
    KDJ14Current = KDJValue(HighPricesCurrent, LowPricesCurrent, ClosePricesCurrent, 14)
    KDJ25Current = KDJValue(HighPricesCurrent, LowPricesCurrent, ClosePricesCurrent, 25)

    data = np.concatenate((OpenPrice, HighPrice,LowPrice,ClosePrice,Volume,
                                    MovingAverage5,MovingAverage10,MovingAverage40,MovingAverage80,
                                    BollingerBand12,BollingerBand20, RSI7,RSI14,RSI25,KDJ7,KDJ14,KDJ25,
                                    OpenPriceCurrent, HighPriceCurrent,LowPriceCurrent,ClosePriceCurrent,VolumeCurrent,
                                    MovingAverage5Current,MovingAverage10Current,MovingAverage40Current,MovingAverage80Current,
                                    BollingerBand12Current,BollingerBand20Current, RSI7Current,RSI14Current,RSI25Current,KDJ7Current,KDJ14Current,KDJ25Current), axis=0)


    return data


def BuildSyntheticNumpyData(PandasArray5, PandasArray15, PandasArray1h, PandasArray4h, PandasArray1d, numpy_link):
    data5 = BuildOneChartData(PandasArray5)
    data15 = BuildOneChartData(PandasArray15)
    data1h = BuildOneChartData(PandasArray1h)
    data4h = BuildOneChartData(PandasArray4h)
    data1d = BuildOneChartData(PandasArray1d)
    SyntheticData = np.concatenate((data5,data15,data1h,
                                    data4h,data1d),axis=0)

    nan_indices = np.argwhere(np.isnan(SyntheticData))

    # 결과 출력
    if len(nan_indices) > 0:
        print(nan_indices)
        print(f"오류검출:"+numpy_link+"에 NaN이 있습니다.")


    np.save(numpy_link,SyntheticData)
    return


def BuildNumpyDatas(crypto_name, adder=0, endsize=30000):
    adder1, adder5, adder15, adder1h, adder4h, adder1d = 287799, 57399, 18999, 4599, 999, 0

    url1, url5, url15, url1h, url4h, url1d = r'G:/CsvStorage/' + crypto_name + '/' + crypto_name + '_1M.csv', \
                                             r'G:/CsvStorage/' + crypto_name + '/' + crypto_name + '_5M.csv', \
                                             r'G:/CsvStorage/' + crypto_name + '/' + crypto_name + '_15M.csv', \
                                             r'G:/CsvStorage/' + crypto_name + '/' + crypto_name + '_1H.csv', \
                                             r'G:/CsvStorage/' + crypto_name + '/' + crypto_name + '_4H.csv', \
                                             r'G:/CsvStorage/' + crypto_name + '/' + crypto_name + '_1D.csv'

    PandasArray5m = TakeCsvData(url5)
    PandasArray15m = TakeCsvData(url15)
    PandasArray1h = TakeCsvData(url1h)
    PandasArray4h = TakeCsvData(url4h)
    PandasArray1d = TakeCsvData(url1d)

    data_size = PandasArray5m.shape[0]

    NumbersOfDataset = int(data_size - LeastNumber2Build - adder5 - adder)
    if endsize ==0:
        NumbersOfDataset = int(data_size - LeastNumber2Build - adder5 - adder)
    elif NumbersOfDataset<endsize :
        NumbersOfDataset = int(data_size - LeastNumber2Build - adder5 - adder)
    else:
        NumbersOfDataset=endsize

    temp15 = 0 + adder // 3
    temp1h = 0 + adder // 12
    temp4h = 0 + adder // 48
    temp1d = 0 + adder // 288
    for temp in range(NumbersOfDataset):

        temp = temp + adder
        starting_x1 = temp + adder5
        ending_x1 = starting_x1 + LeastNumber2Build

        #   if (temp + 1) % 5 == 0:
        #  temp5 += 1

        if (temp + 1) % 15 == 0:
            temp15 += 1

        if (temp + 1) % 60 == 0:
            temp1h += 1

        if (temp + 1) % 240 == 0:
            temp4h += 1

        if (temp + 1) % 1440 == 0:
            temp1d += 1

        # starting_x5 = temp5 + adder5
        # ending_x5 = starting_x5 + LeastNumber2Build

        starting_x15 = temp15 + adder15
        ending_x15 = starting_x15 + LeastNumber2Build

        starting_x1h = temp1h + adder1h
        ending_x1h = starting_x1h + LeastNumber2Build

        starting_x4h = temp4h + adder4h
        ending_x4h = starting_x4h + LeastNumber2Build

        starting_x1d = temp1d + adder1d
        ending_x1d = starting_x1d + LeastNumber2Build

        #   processed_data1 = numpy_data1.iloc[starting_x1:ending_x1]
        #  processed_data1.reset_index(inplace=True)
        # processed_data1.index += 1

        processed_data5 = PandasArray5m.iloc[starting_x1:ending_x1]
        processed_data5.reset_index(inplace=True)
        processed_data5.index += 1

        processed_data15 = PandasArray15m.iloc[starting_x15:ending_x15]
        processed_data15.reset_index(inplace=True)
        processed_data15.index += 1

        processed_data1h = PandasArray1h.iloc[starting_x1h:ending_x1h]
        processed_data1h.reset_index(inplace=True)
        processed_data1h.index += 1

        processed_data4h = PandasArray4h.iloc[starting_x4h:ending_x4h]
        processed_data4h.reset_index(inplace=True)
        processed_data4h.index += 1

        processed_data1d = PandasArray1d.iloc[starting_x1d:ending_x1d]
        processed_data1d.reset_index(inplace=True)
        processed_data1d.index += 1

        parquet_link = ParquetMainRoot + crypto_name + '/' + str(temp + 1) + '.npy'  # 이미지 파일 링크 선정

        BuildSyntheticNumpyData(processed_data5, processed_data15, processed_data1h,
                         processed_data4h, processed_data1d, parquet_link)

def BuildPredictData(crypto_name):

    url1, url5, url15, url1h, url4h, url1d = r'G:/CsvStorage/Trade/' + crypto_name + '_1M.csv', \
                                             r'G:/CsvStorage/Trade/' + crypto_name + '_5M.csv', \
                                             r'G:/CsvStorage/Trade/' + crypto_name + '_15M.csv', \
                                             r'G:/CsvStorage/Trade/' + crypto_name + '_1H.csv', \
                                             r'G:/CsvStorage/Trade/' + crypto_name + '_4H.csv', \
                                             r'G:/CsvStorage/Trade/' + crypto_name + '_1D.csv'

    PandasArray5m = TakeCsvData(url5)
    PandasArray15m = TakeCsvData(url15)
    PandasArray1h = TakeCsvData(url1h)
    PandasArray4h = TakeCsvData(url4h)
    PandasArray1d = TakeCsvData(url1d)

    processed_data5 = PandasArray5m.iloc[-240:]
    processed_data5.reset_index(inplace=True)
    processed_data5.index += 1

    processed_data15 = PandasArray15m.iloc[-240:]
    processed_data15.reset_index(inplace=True)
    processed_data15.index += 1

    processed_data1h = PandasArray1h.iloc[-240:]
    processed_data1h.reset_index(inplace=True)
    processed_data1h.index += 1

    processed_data4h = PandasArray4h.iloc[-240:]
    processed_data4h.reset_index(inplace=True)
    processed_data4h.index += 1

    processed_data1d = PandasArray1d.iloc[-240:]
    processed_data1d.reset_index(inplace=True)
    processed_data1d.index += 1

    parquet_link = './Predict.npy'  # 이미지 파일 링크 선정

    BuildSyntheticNumpyData(processed_data5, processed_data15, processed_data1h,
                            processed_data4h, processed_data1d, parquet_link)



import torch.multiprocessing as mp

if __name__ == '__main__':
    gap =30000
    workers = [mp.Process(target=BuildNumpyDatas, args=('LTCUSDT', gap * (i + 1), gap)) for i in range(11)]

    [w.start() for w in workers]
    [w.join() for w in workers]

    gap =30000
    workers = [mp.Process(target=BuildNumpyDatas, args=('BCHUSDT', gap * (i + 1), gap)) for i in range(11)]

    [w.start() for w in workers]
    [w.join() for w in workers]

    #gap =30000
    #workers = [mp.Process(target=BuildNumpyDatas, args=('BTCUSDT', gap * (i + 1), gap)) for i in range(12)]

    #[w.start() for w in workers]
    #[w.join() for w in workers]
    #gap =30000
    #workers = [mp.Process(target=BuildNumpyDatas, args=('ETHUSDT', gap * (i + 1), gap)) for i in range(12)]

    #[w.start() for w in workers]
    #[w.join() for w in workers]

   # gap = 30000
   # workers = [mp.Process(target=BuildNumpyDatas, args=('SOLUSDT', gap * (i + 1), gap)) for i in range(9)]

    #[w.start() for w in workers]
  #  [w.join() for w in workers]

  #  gap =30000
  #  workers = [mp.Process(target=BuildNumpyDatas, args=('XRPUSDT', gap * (i + 1), gap)) for i in range(12)]

  #  [w.start() for w in workers]
   # [w.join() for w in workers]
