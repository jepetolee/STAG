import sys
import numpy as np

from .ModifyCsv import *
from mpl_finance import candlestick2_ohlc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import trange
import cv2
from PIL import Image

sys.path.append('..')

LeastNumber2Build = 200  # 0~199
IMGFileMainRoot = 'G:/ImgDataStorage/'
ParquetMainRoot = 'G:/ParquetDataStorage/'


def build_single_candlestick_images(crypto_name, adder, interval, url):
    numpy_data = TakeCsvData(url)
    data_size = numpy_data.shape[0]
    NumbersOfDataset = int(data_size - LeastNumber2Build - adder)
    for starting_x in trange(NumbersOfDataset):
        starting_x += adder
        # building files to sequence
        ending_x = LeastNumber2Build + starting_x
        processed_data = numpy_data.iloc[starting_x:ending_x]
        processed_data.reset_index(inplace=True)
        processed_data.index += 1

        image_link = IMGFileMainRoot + crypto_name + '/' + interval + '/' + str(starting_x + 1) + '.jpg'  # 이미지 파일 링크 선정
        BuildNSaveImage(processed_data, image_link)


def build_parquet_datas(crypto_name, adder=0):
    adder1,adder5,adder15, adder1h, adder4h, adder1d = 287799,57399,18999, 4599, 999, 0
    url1,url5,url15, url1h, url4h, url1d = r'G:/CsvStorage/' + crypto_name + '/' + crypto_name + '_1M.csv', \
                                    r'G:/CsvStorage/' + crypto_name + '/' + crypto_name + '_5M.csv', \
                                 r'G:/CsvStorage/' + crypto_name + '/' + crypto_name + '_15M.csv', \
                                 r'G:/CsvStorage/' + crypto_name + '/' + crypto_name + '_1H.csv', \
                                 r'G:/CsvStorage/' + crypto_name + '/' + crypto_name + '_4H.csv', \
                                 r'G:/CsvStorage/' + crypto_name + '/' + crypto_name + '_1D.csv'
    numpy_data1 = TakeCsvData(url1)
    numpy_data5 = TakeCsvData(url5)
    numpy_data15 = TakeCsvData(url15)
    numpy_data1h = TakeCsvData(url1h)
    numpy_data4h = TakeCsvData(url4h)
    numpy_data1d = TakeCsvData(url1d)

    data_size = numpy_data1.shape[0]
    NumbersOfDataset = int(data_size - LeastNumber2Build - adder1 - adder)

    temp5 = 0 + adder // 5
    temp15 = 0 + adder // 15
    temp1h = 0 + adder // 60
    temp4h = 0 + adder // 240
    temp1d = 0 + adder // 1440
    for temp in trange(NumbersOfDataset):

        temp = temp + adder
        starting_x1 = temp + adder1
        ending_x1 = starting_x1 + LeastNumber2Build

        if (temp + 1) % 5 == 0:
            temp5 += 1

        if (temp + 1) % 15 == 0:
            temp15 += 1

        if (temp + 1) % 60 == 0:
            temp1h += 1

        if (temp + 1) % 240 == 0:
            temp4h += 1

        if (temp + 1) % 1440 == 0:
            temp1d += 1

        starting_x5 = temp5 + adder5
        ending_x5 = starting_x5 + LeastNumber2Build

        starting_x15 = temp15 + adder15
        ending_x15 = starting_x15 + LeastNumber2Build

        starting_x1h = temp1h + adder1h
        ending_x1h = starting_x1h + LeastNumber2Build

        starting_x4h = temp4h + adder4h
        ending_x4h = starting_x4h + LeastNumber2Build

        starting_x1d = temp1d + adder1d
        ending_x1d = starting_x1d + LeastNumber2Build

        processed_data1 = numpy_data1.iloc[starting_x1:ending_x1]
        processed_data1.reset_index(inplace=True)
        processed_data1.index += 1

        processed_data5 = numpy_data5.iloc[starting_x5:ending_x5]
        processed_data5.reset_index(inplace=True)
        processed_data5.index += 1

        processed_data15 = numpy_data15.iloc[starting_x15:ending_x15]
        processed_data15.reset_index(inplace=True)
        processed_data15.index += 1

        processed_data1h = numpy_data1h.iloc[starting_x1h:ending_x1h]
        processed_data1h.reset_index(inplace=True)
        processed_data1h.index += 1

        processed_data4h = numpy_data4h.iloc[starting_x4h:ending_x4h]
        processed_data4h.reset_index(inplace=True)
        processed_data4h.index += 1

        processed_data1d = numpy_data1d.iloc[starting_x1d:ending_x1d]
        processed_data1d.reset_index(inplace=True)
        processed_data1d.index += 1

        parquet_link = ParquetMainRoot + crypto_name + '/' + str(temp + 1) + '.parquet'  # 이미지 파일 링크 선정

        BuildNewStyleOfData(processed_data1,processed_data5,processed_data15, processed_data1h, processed_data4h, processed_data1d, parquet_link)


def BuildNewStyleOfData(numpy_data1,numpy_data5, numpy_data15, numpy_data1h, numpy_data4h, numpy_data1d, parquet_link):
    data = pd.DataFrame()

    ClosePrices = numpy_data1['ClosePrice'].copy()
    HighPrice = numpy_data1['HighPrice'].copy()
    LowPrice = numpy_data1['LowPrice'].copy()
    OpenPrice = numpy_data1['OpenPrice'].copy()

    data['ClosePrice'] = ClosePrices / ClosePrices.abs().max()
    data['HighPrice'] = HighPrice / HighPrice.abs().max()
    data['LowPrice'] = LowPrice / LowPrice.abs().max()
    data['OpenPrice'] = OpenPrice / OpenPrice.abs().max()
    TwentyAvg = ClosePrices.rolling(window=20).mean()
    data['TwentyAvg'] = TwentyAvg / TwentyAvg.abs().max()
    FiftyAvg = ClosePrices.rolling(window=50).mean()
    data['FiftyAvg'] = FiftyAvg / FiftyAvg.abs().max()
    HundredAvg = ClosePrices.rolling(window=100).mean()
    data['HundredAvg'] = HundredAvg / HundredAvg.abs().max()
    data['Volume'] = numpy_data1['Volume'] / numpy_data1['Volume'].abs().max()
    TwentyStandardDeviation = ClosePrices.rolling(window=20).std()
    BollingerBandUpper = TwentyAvg + 2 * TwentyStandardDeviation
    BollingerBandLower = TwentyAvg - 2 * TwentyStandardDeviation
    data['BollingerBandUpper'] = BollingerBandUpper / BollingerBandUpper.abs().max()
    data['BollingerBandLower'] = BollingerBandLower / BollingerBandLower.abs().max()
    variance = ClosePrices - ClosePrices.shift(1)
    rise_width = variance.where(variance >= 0, 0)
    degrade_width = variance.where(variance < 0, 0).abs()
    AverageUp = rise_width.ewm(alpha=1 / 14, min_periods=14).mean()
    AverageDown = degrade_width.ewm(alpha=1 / 14, min_periods=14).mean()
    RSI = AverageUp / (AverageUp + AverageDown)
    data['RSI'] = RSI / RSI.abs().max()
    HighPrices = numpy_data1['HighPrice'].copy()
    HighInDays = HighPrices.rolling(window=9, min_periods=1).max()
    LowPrices = numpy_data1['LowPrice'].copy()
    LowInDays = LowPrices.rolling(window=9, min_periods=1).max()
    PeriodRSV = 100 * (ClosePrices - LowPrices) / (HighInDays - LowInDays)
    PercentK = PeriodRSV.ewm(span=3).mean()
    PercentD = PercentK.ewm(span=3).mean()
    PercentJ = 3 * PercentK - 2 * PercentD
    data['PercentK'] = PercentK / PercentK.abs().max()
    data['PercentD'] = PercentD / PercentD.abs().max()
    data['PercentJ'] = PercentJ / PercentJ.abs().max()

    ClosePrices = numpy_data5['ClosePrice'].copy()
    HighPrice = numpy_data5['HighPrice'].copy()
    LowPrice = numpy_data5['LowPrice'].copy()
    OpenPrice = numpy_data5['OpenPrice'].copy()

    data['ClosePrice5'] = ClosePrices / ClosePrices.abs().max()
    data['HighPrice5'] = HighPrice / HighPrice.abs().max()
    data['LowPrice5'] = LowPrice / LowPrice.abs().max()
    data['OpenPrice5'] = OpenPrice / OpenPrice.abs().max()
    TwentyAvg = ClosePrices.rolling(window=20).mean()
    data['TwentyAvg5'] = TwentyAvg / TwentyAvg.abs().max()
    FiftyAvg = ClosePrices.rolling(window=50).mean()
    data['FiftyAvg5'] = FiftyAvg / FiftyAvg.abs().max()
    HundredAvg = ClosePrices.rolling(window=100).mean()
    data['HundredAvg5'] = HundredAvg / HundredAvg.abs().max()
    data['Volume5'] = numpy_data5['Volume'] / numpy_data5['Volume'].abs().max()
    TwentyStandardDeviation = ClosePrices.rolling(window=20).std()
    BollingerBandUpper = TwentyAvg + 2 * TwentyStandardDeviation
    BollingerBandLower = TwentyAvg - 2 * TwentyStandardDeviation
    data['BollingerBandUpper5'] = BollingerBandUpper / BollingerBandUpper.abs().max()
    data['BollingerBandLower5'] = BollingerBandLower / BollingerBandLower.abs().max()
    variance = ClosePrices - ClosePrices.shift(1)
    rise_width = variance.where(variance >= 0, 0)
    degrade_width = variance.where(variance < 0, 0).abs()
    AverageUp = rise_width.ewm(alpha=1 / 14, min_periods=14).mean()
    AverageDown = degrade_width.ewm(alpha=1 / 14, min_periods=14).mean()
    RSI = AverageUp / (AverageUp + AverageDown)
    data['RSI5'] = RSI / RSI.abs().max()
    HighPrices = numpy_data5['HighPrice'].copy()
    HighInDays = HighPrices.rolling(window=9, min_periods=1).max()
    LowPrices = numpy_data5['LowPrice'].copy()
    LowInDays = LowPrices.rolling(window=9, min_periods=1).max()
    PeriodRSV = 100 * (ClosePrices - LowPrices) / (HighInDays - LowInDays)
    PercentK = PeriodRSV.ewm(span=3).mean()
    PercentD = PercentK.ewm(span=3).mean()
    PercentJ = 3 * PercentK - 2 * PercentD
    data['PercentK5'] = PercentK / PercentK.abs().max()
    data['PercentD5'] = PercentD / PercentD.abs().max()
    data['PercentJ5'] = PercentJ / PercentJ.abs().max()

    ClosePrices = numpy_data15['ClosePrice'].copy()
    HighPrice = numpy_data15['HighPrice'].copy()
    LowPrice = numpy_data15['LowPrice'].copy()
    OpenPrice = numpy_data15['OpenPrice'].copy()

    data['ClosePrice15'] = ClosePrices / ClosePrices.abs().max()
    data['HighPrice15'] = HighPrice / HighPrice.abs().max()
    data['LowPrice15'] = LowPrice / LowPrice.abs().max()
    data['OpenPrice15'] = OpenPrice / OpenPrice.abs().max()
    TwentyAvg = ClosePrices.rolling(window=20).mean()
    data['TwentyAvg15'] = TwentyAvg / TwentyAvg.abs().max()
    FiftyAvg = ClosePrices.rolling(window=50).mean()
    data['FiftyAvg15'] = FiftyAvg / FiftyAvg.abs().max()
    HundredAvg = ClosePrices.rolling(window=100).mean()
    data['HundredAvg15'] = HundredAvg / HundredAvg.abs().max()
    data['Volume15'] = numpy_data15['Volume'] / numpy_data15['Volume'].abs().max()
    TwentyStandardDeviation = ClosePrices.rolling(window=20).std()
    BollingerBandUpper = TwentyAvg + 2 * TwentyStandardDeviation
    BollingerBandLower = TwentyAvg - 2 * TwentyStandardDeviation
    data['BollingerBandUpper15'] = BollingerBandUpper / BollingerBandUpper.abs().max()
    data['BollingerBandLower15'] = BollingerBandLower / BollingerBandLower.abs().max()
    variance = ClosePrices - ClosePrices.shift(1)
    rise_width = variance.where(variance >= 0, 0)
    degrade_width = variance.where(variance < 0, 0).abs()
    AverageUp = rise_width.ewm(alpha=1 / 14, min_periods=14).mean()
    AverageDown = degrade_width.ewm(alpha=1 / 14, min_periods=14).mean()
    RSI = AverageUp / (AverageUp + AverageDown)
    data['RSI15'] = RSI / RSI.abs().max()
    HighPrices = numpy_data15['HighPrice'].copy()
    HighInDays = HighPrices.rolling(window=9, min_periods=1).max()
    LowPrices = numpy_data15['LowPrice'].copy()
    LowInDays = LowPrices.rolling(window=9, min_periods=1).max()
    PeriodRSV = 100 * (ClosePrices - LowPrices) / (HighInDays - LowInDays)
    PercentK = PeriodRSV.ewm(span=3).mean()
    PercentD = PercentK.ewm(span=3).mean()
    PercentJ = 3 * PercentK - 2 * PercentD
    data['PercentK15'] = PercentK / PercentK.abs().max()
    data['PercentD15'] = PercentD / PercentD.abs().max()
    data['PercentJ15'] = PercentJ / PercentJ.abs().max()

    ClosePrices = numpy_data1h['ClosePrice'].copy()
    HighPrice = numpy_data1h['HighPrice'].copy()
    LowPrice = numpy_data1h['LowPrice'].copy()
    OpenPrice = numpy_data1h['OpenPrice'].copy()

    data['ClosePrice1h'] = ClosePrices / ClosePrices.abs().max()
    data['HighPrice1h'] = HighPrice / HighPrice.abs().max()
    data['LowPrice1h'] = LowPrice / LowPrice.abs().max()
    data['OpenPrice1h'] = OpenPrice / OpenPrice.abs().max()
    TwentyAvg = ClosePrices.rolling(window=20).mean()
    data['TwentyAvg1h'] = TwentyAvg / TwentyAvg.abs().max()
    FiftyAvg = ClosePrices.rolling(window=50).mean()
    data['FiftyAvg1h'] = FiftyAvg / FiftyAvg.abs().max()
    HundredAvg = ClosePrices.rolling(window=100).mean()
    data['HundredAvg1h'] = HundredAvg / HundredAvg.abs().max()
    data['Volume1h'] = numpy_data1h['Volume'] / numpy_data1h['Volume'].abs().max()
    TwentyStandardDeviation = ClosePrices.rolling(window=20).std()
    BollingerBandUpper = TwentyAvg + 2 * TwentyStandardDeviation
    BollingerBandLower = TwentyAvg - 2 * TwentyStandardDeviation
    data['BollingerBandUpper1h'] = BollingerBandUpper / BollingerBandUpper.abs().max()
    data['BollingerBandLower1h'] = BollingerBandLower / BollingerBandLower.abs().max()
    variance = ClosePrices - ClosePrices.shift(1)
    rise_width = variance.where(variance >= 0, 0)
    degrade_width = variance.where(variance < 0, 0).abs()
    AverageUp = rise_width.ewm(alpha=1 / 14, min_periods=14).mean()
    AverageDown = degrade_width.ewm(alpha=1 / 14, min_periods=14).mean()
    RSI = AverageUp / (AverageUp + AverageDown)
    data['RSI1h'] = RSI / RSI.abs().max()
    HighPrices = numpy_data1h['HighPrice'].copy()
    HighInDays = HighPrices.rolling(window=9, min_periods=1).max()
    LowPrices = numpy_data1h['LowPrice'].copy()
    LowInDays = LowPrices.rolling(window=9, min_periods=1).max()
    PeriodRSV = 100 * (ClosePrices - LowPrices) / (HighInDays - LowInDays)
    PercentK = PeriodRSV.ewm(span=3).mean()
    PercentD = PercentK.ewm(span=3).mean()
    PercentJ = 3 * PercentK - 2 * PercentD
    data['PercentK1h'] = PercentK / PercentK.abs().max()
    data['PercentD1h'] = PercentD / PercentD.abs().max()
    data['PercentJ1h'] = PercentJ / PercentJ.abs().max()


    ClosePrices = numpy_data4h['ClosePrice'].copy()
    HighPrice = numpy_data4h['HighPrice'].copy()
    LowPrice = numpy_data4h['LowPrice'].copy()
    OpenPrice = numpy_data4h['OpenPrice'].copy()

    data['ClosePrice4h'] = ClosePrices / ClosePrices.abs().max()
    data['HighPrice4h'] = HighPrice / HighPrice.abs().max()
    data['LowPrice4h'] = LowPrice / LowPrice.abs().max()
    data['OpenPrice4h'] = OpenPrice / OpenPrice.abs().max()
    TwentyAvg = ClosePrices.rolling(window=20).mean()
    data['TwentyAvg4h'] = TwentyAvg / TwentyAvg.abs().max()
    FiftyAvg = ClosePrices.rolling(window=50).mean()
    data['FiftyAvg4h'] = FiftyAvg / FiftyAvg.abs().max()
    HundredAvg = ClosePrices.rolling(window=100).mean()
    data['HundredAvg4h'] = HundredAvg / HundredAvg.abs().max()
    data['Volume4h'] = numpy_data4h['Volume'] / numpy_data4h['Volume'].abs().max()
    TwentyStandardDeviation = ClosePrices.rolling(window=20).std()
    BollingerBandUpper = TwentyAvg + 2 * TwentyStandardDeviation
    BollingerBandLower = TwentyAvg - 2 * TwentyStandardDeviation
    data['BollingerBandUpper4h'] = BollingerBandUpper / BollingerBandUpper.abs().max()
    data['BollingerBandLower4h'] = BollingerBandLower / BollingerBandLower.abs().max()
    variance = ClosePrices - ClosePrices.shift(1)
    rise_width = variance.where(variance >= 0, 0)
    degrade_width = variance.where(variance < 0, 0).abs()
    AverageUp = rise_width.ewm(alpha=1 / 14, min_periods=14).mean()
    AverageDown = degrade_width.ewm(alpha=1 / 14, min_periods=14).mean()
    RSI = AverageUp / (AverageUp + AverageDown)
    data['RSI4h'] = RSI / RSI.abs().max()
    HighPrices = numpy_data4h['HighPrice'].copy()
    HighInDays = HighPrices.rolling(window=9, min_periods=1).max()
    LowPrices = numpy_data4h['LowPrice'].copy()
    LowInDays = LowPrices.rolling(window=9, min_periods=1).max()
    PeriodRSV = 100 * (ClosePrices - LowPrices) / (HighInDays - LowInDays)
    PercentK = PeriodRSV.ewm(span=3).mean()
    PercentD = PercentK.ewm(span=3).mean()
    PercentJ = 3 * PercentK - 2 * PercentD
    data['PercentK4h'] = PercentK / PercentK.abs().max()
    data['PercentD4h'] = PercentD / PercentD.abs().max()
    data['PercentJ4h'] = PercentJ / PercentJ.abs().max()


    ClosePrices = numpy_data1d['ClosePrice'].copy()
    HighPrice = numpy_data1d['HighPrice'].copy()
    LowPrice = numpy_data1d['LowPrice'].copy()
    OpenPrice = numpy_data1d['OpenPrice'].copy()

    data['ClosePrice1d'] = ClosePrices / ClosePrices.abs().max()
    data['HighPrice1d'] = HighPrice / HighPrice.abs().max()
    data['LowPrice1d'] = LowPrice / LowPrice.abs().max()
    data['OpenPrice1d'] = OpenPrice / OpenPrice.abs().max()
    data['ClosePrice1d'] = ClosePrices / ClosePrices.abs().max()
    TwentyAvg = ClosePrices.rolling(window=20).mean()
    data['TwentyAvg1d'] = TwentyAvg / TwentyAvg.abs().max()
    FiftyAvg = ClosePrices.rolling(window=50).mean()
    data['FiftyAvg1d'] = FiftyAvg / FiftyAvg.abs().max()
    HundredAvg = ClosePrices.rolling(window=100).mean()
    data['HundredAvg1d'] = HundredAvg / HundredAvg.abs().max()
    data['Volume1d'] = numpy_data1d['Volume'] / numpy_data1d['Volume'].abs().max()
    TwentyStandardDeviation = ClosePrices.rolling(window=20).std()
    BollingerBandUpper = TwentyAvg + 2 * TwentyStandardDeviation
    BollingerBandLower = TwentyAvg - 2 * TwentyStandardDeviation
    data['BollingerBandUpper1d'] = BollingerBandUpper / BollingerBandUpper.abs().max()
    data['BollingerBandLower1d'] = BollingerBandLower / BollingerBandLower.abs().max()
    variance = ClosePrices - ClosePrices.shift(1)
    rise_width = variance.where(variance >= 0, 0)
    degrade_width = variance.where(variance < 0, 0).abs()
    AverageUp = rise_width.ewm(alpha=1 / 14, min_periods=14).mean()
    AverageDown = degrade_width.ewm(alpha=1 / 14, min_periods=14).mean()
    RSI = AverageUp / (AverageUp + AverageDown)
    data['RSI1d'] = RSI / RSI.abs().max()
    HighPrices = numpy_data1d['HighPrice'].copy()
    HighInDays = HighPrices.rolling(window=9, min_periods=1).max()
    LowPrices = numpy_data1d['LowPrice'].copy()
    LowInDays = LowPrices.rolling(window=9, min_periods=1).max()
    PeriodRSV = 100 * (ClosePrices - LowPrices) / (HighInDays - LowInDays)
    PercentK = PeriodRSV.ewm(span=3).mean()
    PercentD = PercentK.ewm(span=3).mean()
    PercentJ = 3 * PercentK - 2 * PercentD
    data['PercentK1d'] = PercentK / PercentK.abs().max()
    data['PercentD1d'] = PercentD / PercentD.abs().max()
    data['PercentJ1d'] = PercentJ / PercentJ.abs().max()

    data = data.replace(np.nan, 0)

    data.to_parquet(parquet_link)


def BuildNSaveImage(numpy_data, image_link):
    # build grid spec pyplot
    chart_figure = plt.figure(figsize=(5, 4))
    chart_grid = gridspec.GridSpec(5, 4)
    chart_figure.subplots_adjust(wspace=0.2, hspace=0.2)

    axes = list()
    axes.append(plt.subplot(chart_grid[0, :]))
    for i in range(4):
        axes.append(plt.subplot(chart_grid[i + 1, :], sharex=axes[0]))

    '''
     Added Ones:
      1. Candlesticks with Moving Average line ( 20 ,50, 100 ,200)
      2. Volumes 
      3. Relative Strength Indicator (14)
      4. Bollinger Band (20 2)
      5. Stochastic Indicator KDJ (9,3,3)
    '''

    candlestick2_ohlc(axes[0], numpy_data['OpenPrice'], numpy_data['HighPrice'], numpy_data['LowPrice'],
                      numpy_data['ClosePrice'], width=1, colorup='g', colordown='r')
    ClosePrices = numpy_data['ClosePrice'].copy()

    TwentyMovingAverage = ClosePrices.rolling(window=20).mean()
    axes[0].plot(numpy_data.index, TwentyMovingAverage, label='Moving Average 20')
    axes[0].plot(numpy_data.index, ClosePrices.rolling(window=50).mean(), label='Moving Average 50')
    axes[0].plot(numpy_data.index, ClosePrices.rolling(window=100).mean(), label='Moving Average 100')
    axes[0].plot(numpy_data.index, ClosePrices.rolling(window=200).mean(), label='Moving Average 200')

    axes[1].bar(numpy_data.index, numpy_data['Volume'], color='k', width=0.8, align='center')

    candlestick2_ohlc(axes[2], numpy_data['OpenPrice'], numpy_data['HighPrice'], numpy_data['LowPrice'],
                      numpy_data['ClosePrice'], width=1, colorup='g', colordown='r')
    TwentyStandardDeviation = ClosePrices.rolling(window=20).std()
    BollingerBandUpper = TwentyMovingAverage + 2 * TwentyStandardDeviation
    BollingerBandLower = TwentyMovingAverage - 2 * TwentyStandardDeviation
    axes[2].plot(numpy_data.index, TwentyMovingAverage, 'y', label='Moving Average 20')
    axes[2].plot(numpy_data.index, BollingerBandUpper, 'k', label='BollingerBandUpper 20 +2')
    axes[2].plot(numpy_data.index, BollingerBandLower, 'b', label='BollingerBandLower 20 -2')

    variance = ClosePrices - ClosePrices.shift(1)
    rise_width = variance.where(variance >= 0, 0)
    degrade_width = variance.where(variance < 0, 0).abs()
    AverageUp = rise_width.ewm(alpha=1 / 14, min_periods=14).mean()
    AverageDown = degrade_width.ewm(alpha=1 / 14, min_periods=14).mean()
    RSI = AverageUp / (AverageUp + AverageDown) * 100
    axes[3].plot(numpy_data.index, RSI, label='RSI 14')

    HighPrices = numpy_data['HighPrice'].copy()
    HighInDays = HighPrices.rolling(window=9, min_periods=1).max()
    LowPrices = numpy_data['LowPrice'].copy()
    LowInDays = LowPrices.rolling(window=9, min_periods=1).max()
    PeriodRSV = 100 * (ClosePrices - LowPrices) / (HighInDays - LowInDays)
    PercentK = PeriodRSV.ewm(span=3).mean()
    PercentD = PercentK.ewm(span=3).mean()
    PercentJ = 3 * PercentK - 2 * PercentD
    axes[4].plot(numpy_data.index, PercentK, 'y', label='%K')
    axes[4].plot(numpy_data.index, PercentD, 'c', label='%D')
    axes[4].plot(numpy_data.index, PercentJ, 'k', label='%J')

    plt.savefig(image_link, dpi=300)
    plt.close('all')
    return


def crop(image):  # 인수는 이미지의 상대 경로
    # 이미지를 읽어들인다.
    img = cv2.imread(image)

    # 주위 부분을 강제적으로 트리밍
    h, w = img.shape[:2]
    h1, h2 = int(h * 0.05), int(h * 0.95)
    w1, w2 = int(w * 0.05), int(w * 0.95)
    img = img[h1: h2, w1: w2]
    # cv2.imshow('img', img)

    # Grayscale으로 변환 (흑백 이미지로 변환)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)

    # 색 공간을 이진화한다.
    img2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow('img2', img2)

    # 윤곽을 추출한다.
    contours = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    # 윤곽의 좌표를 리스트에 대입한다.
    x1 = []  # x좌표의 최소값
    y1 = []  # y좌표의 최소값
    x2 = []  # x좌표의 최대값
    y2 = []  # y좌표의 최대값
    for i in range(1, len(contours)):  # i = 1 는 이미지 전체의 외곽이되므로 카운트에 포함시키지 않는다.
        ret = cv2.boundingRect(contours[i])
        x1.append(ret[0])
        y1.append(ret[1])
        x2.append(ret[0] + ret[2])
        y2.append(ret[1] + ret[3])

    # 외곽의 첫 번째 외곽을 오려냄
    x1_min = min(x1)
    y1_min = min(y1)
    x2_max = max(x2)
    y2_max = max(y2)
    cv2.rectangle(img, (x1_min, y1_min), (x2_max, y2_max), (0, 255, 0), 3)

    crop_img = img2[y1_min:y2_max, x1_min:x2_max]
    color_coverted = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

    return color_coverted


def synthesize_image(symbol, adder):
    fifteenminute_link = 'G:/CsvStorage/' + symbol + '/' + symbol + '_15M.csv'
    print(fifteenminute_link)
    fifteenminute_data = TakeCsvData(fifteenminute_link)
    fifteenminute_data_size = int(fifteenminute_data.shape[0] - LeastNumber2Build - adder)

    basic_image_root = 'G:/ImgDataStorage/' + symbol + '/'

    onehour_starter, fourhour_starter, oneday_starter = adder // 4, adder // 16, adder // 96
    for starting_x in trange(fifteenminute_data_size - 19000):
        starting_x += 1 + adder
        if starting_x % 4 == 1:
            onehour_starter += 1
            if starting_x % 16 == 1:
                fourhour_starter += 1
                if starting_x % 96 == 1:
                    oneday_starter += 1

        oneday_root = basic_image_root + '1D/' + str(oneday_starter) + '.jpg'
        oneday_image = crop(oneday_root)
        oneday_data = Image.fromarray(oneday_image)

        fourhour_root = basic_image_root + '4H/' + str(fourhour_starter + 1000) + '.jpg'
        fourhour_image = crop(fourhour_root)
        fourhour_data = Image.fromarray(fourhour_image)

        onehour_root = basic_image_root + '1H/' + str(onehour_starter + 4600) + '.jpg'
        onehour_image = crop(onehour_root)
        onehour_data = Image.fromarray(onehour_image)

        fifteenminute_root = basic_image_root + '15M/' + str(starting_x + 19000) + '.jpg'
        fifteenminute_image = crop(fifteenminute_root)
        fifteenminute_data = Image.fromarray(fifteenminute_image)

        COMBINED_root = basic_image_root + 'New/' + str(starting_x) + '.jpg'

        COMBINED_image = Image.new('RGB', (2 * fourhour_data.size[0], 2 * fourhour_data.size[1]))

        COMBINED_image.paste(oneday_data, (0, 0))
        COMBINED_image.paste(fourhour_data, (fourhour_data.size[0], 0))
        COMBINED_image.paste(onehour_data, (0, fourhour_data.size[1]))
        COMBINED_image.paste(fifteenminute_data, (fourhour_data.size[0], fourhour_data.size[1]))

        COMBINED_image.save(COMBINED_root, 'PNG')
    return

#build_parquet_datas('BTCUSDT',adder=1600000)
# synthesize_image('BTCUSDT', 0)
# ADAUSDT BTCUSDT ,DOGEUSDT,ETHUSDT,ETCUSDT,XRPUSDT
