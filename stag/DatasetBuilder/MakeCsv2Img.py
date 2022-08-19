import sys
from stag.DatasetBuilder.ModifyCsv import *
from mpl_finance import candlestick2_ohlc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import trange

sys.path.append('..')

LeastNumber2Build = 199  # 0~199
IMGFileMainRoot = ''


def ImagedataSetBuilder(url):
    numpy_data = TakeCsvData(url)

    data_size = numpy_data.shape[0]
    NumbersOfDataset = int(data_size - LeastNumber2Build)

    for starting_x in trange(NumbersOfDataset):
        # building files to sequence
        ending_x = LeastNumber2Build + starting_x
        processed_data = numpy_data.iloc[starting_x:ending_x]
        processed_data.reset_index(inplace=True)
        processed_data.index += 1

        image_link = IMGFileMainRoot + str(starting_x + 1) + 'jpg'  # 이미지 파일 링크 선정
        BuildNSaveImage(processed_data, image_link)


def BuildNSaveImage(numpy_data, image_link):
    # build grid spec pyplot
    chart_figure = plt.figure(50, 40)
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
                      numpy_data['ClosePrice'], width=1, colorup='r', colordown='b')
    ClosePrices = numpy_data['ClosePrice'].copy()

    TwentyMovingAverage = ClosePrices.rolling(window=20).mean()
    axes[0].plot(numpy_data.index, TwentyMovingAverage, label='Moving Average 20')
    axes[0].plot(numpy_data.index, ClosePrices.rolling(window=50).mean(), label='Moving Average 50')
    axes[0].plot(numpy_data.index, ClosePrices.rolling(window=100).mean(), label='Moving Average 100')
    axes[0].plot(numpy_data.index, ClosePrices.rolling(window=200).mean(), label='Moving Average 200')

    axes[1].bar(numpy_data.index, numpy_data['Volume'], color='k', width=0.8, align='center')

    candlestick2_ohlc(axes[2], numpy_data['OpenPrice'], numpy_data['HighPrice'], numpy_data['LowPrice'],
                      numpy_data['ClosePrice'], width=1, colorup='r', colordown='b')
    TwentyStandardDeviation = ClosePrices.rolling(window=20).std()
    BollingerBandUpper = TwentyMovingAverage + 2 * TwentyStandardDeviation
    BollingerBandLower = TwentyMovingAverage - 2 * TwentyStandardDeviation
    axes[2].plot(numpy_data.index, TwentyMovingAverage, 'r', label='Moving Average 20')
    axes[2].plot(numpy_data.index, BollingerBandUpper, 'g', label='BollingerBandUpper 20 +2')
    axes[2].plot(numpy_data.index, BollingerBandLower, 'b', label='BollingerBandLower 20 -2')

    variance = ClosePrices - ClosePrices.shift(1)
    rise_width = variance.where(variance > 0, 0)
    degrade_width = variance.where(variance < 0, 0)
    AverageUp = rise_width.ewm(aplha=1 / 14, min_periods=14).mean()
    AverageDown = degrade_width.ewm(aplha=1 / 14, min_periods=14).mean()
    RSI = AverageUp / (AverageUp + AverageDown) * 100
    axes[3].bar(numpy_data.index, RSI, label='RSI 14')

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

    plt.savefig(image_link, dpi=100)
    plt.close('all')
    return
