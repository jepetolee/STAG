START_DATE = "2018-01-01"
TIME_ONE_DAY = '1d'
TIME_ONE_HOUR = '1h'
TIME_FOUR_HOUR = '4h'
TIME_FIFTEEN_MINUTE = '15m'
TIME_FIVE_MINUTE = '5m'

import pandas as pd
from binance.client import Client
import numpy as np

def BuildCsv(time_interval, crypto_symbol, URL,start_date=START_DATE):
    client = Client(api_key='', api_secret='')
    k_lines = client.futures_historical_klines(symbol=crypto_symbol, interval=time_interval,
                                               start_str=start_date, limit=1000)
    current = client.get_historical_klines(symbol=crypto_symbol, interval=time_interval,
                                               start_str=start_date, limit=1000)
    dataframe = pd.DataFrame(k_lines).to_numpy()
    dataframe_current = pd.DataFrame(current).to_numpy()

    if dataframe.shape[0]>= dataframe_current.shape[0]:
        dataframe = dataframe[dataframe.shape[0]-dataframe_current.shape[0]:]
    else:
        dataframe_current = dataframe_current[dataframe_current.shape[0]-dataframe.shape[0]:]
    new_data = np.concatenate([dataframe, dataframe_current], axis=1)

    merged_column = ["0","1","2","3","4","5","6","7","8","9","10","11","2_0", "2_1", "2_2", "2_3", "2_4", "2_5", "2_6","2_7","2_8","2_9","2_10","2_11"]

    merged = pd.DataFrame(new_data,columns=merged_column)

    merged.to_csv(URL)
    return


def UpdateCsv(time_interval, crypto_symbol, URL):
    client = Client(api_key='', api_secret='')

    try:
        csv_data = pd.read_csv(URL)
        unreadable_time_data = csv_data['0'].copy().to_numpy()
        readable_time_data = unreadable_time_data.astype('datetime64[ms]')

        k_lines = client.futures_historical_klines(symbol=crypto_symbol, interval=time_interval,
                                                   start_str=str(readable_time_data[-1]), limit=1000)
        dataframe = pd.DataFrame(k_lines)
        dataframe.to_csv('./appender.csv')

        dataframe = pd.read_csv('./appender.csv')

        csv_data.drop(['Unnamed: 0'], axis=1, inplace=True)
        csv_data = csv_data.loc[0:csv_data.index[-2]]
        dataframe.drop(['Unnamed: 0'], axis=1, inplace=True)

        csv_data = pd.concat([csv_data, dataframe], ignore_index=True)
        csv_data.to_csv(URL)
        return

    except FileNotFoundError:
        print(f"Csv file was not found in {URL}")
        return


def TakeCsvData(URL):
    try:
        csv_data = pd.read_csv(URL)

        unreadable_time_data = csv_data['0'].copy()
        readable_time_data = unreadable_time_data.to_numpy().astype('datetime64[ms]')

        csv_data = csv_data[['1', '2', '3', '4', '5','2_1','2_2','2_3','2_4','2_5']]
        csv_data.columns = ["OpenPrice", "HighPrice","LowPrice","ClosePrice","Volume","OpenPriceCurrent", "HighPriceCurrent","LowPriceCurrent","ClosePriceCurrent","VolumeCurrent"]
        csv_data = csv_data.set_index(readable_time_data)

        return csv_data

    except FileNotFoundError:
        print(f"Csv file was not found in {URL}")
        return


'''
DataName = 'BCHUSDT'
BuildCsv(TIME_FIVE_MINUTE,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_5M.csv')
DataName = 'LTCUSDT'
BuildCsv(TIME_ONE_DAY,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_1D.csv')
BuildCsv(TIME_FOUR_HOUR,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_4H.csv')
BuildCsv(TIME_ONE_HOUR,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_1H.csv')
BuildCsv(TIME_FIFTEEN_MINUTE,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_15M.csv')
BuildCsv(TIME_FIVE_MINUTE,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_5M.csv')


DataName = 'BCHUSDT'
BuildCsv(TIME_ONE_DAY,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_1D.csv')
BuildCsv(TIME_FOUR_HOUR,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_4H.csv')
BuildCsv(TIME_ONE_HOUR,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_1H.csv')
BuildCsv(TIME_FIFTEEN_MINUTE,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_15M.csv')
BuildCsv(TIME_FIVE_MINUTE,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_5M.csv')
DataName = 'BTCUSDT'
BuildCsv(TIME_ONE_DAY,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_1D.csv')
BuildCsv(TIME_FOUR_HOUR,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_4H.csv')
BuildCsv(TIME_ONE_HOUR,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_1H.csv')
BuildCsv(TIME_FIFTEEN_MINUTE,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_15M.csv')
BuildCsv(TIME_FIVE_MINUTE,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_5M.csv')

DataName = 'ETHUSDT'
BuildCsv(TIME_ONE_DAY,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_1D.csv')
BuildCsv(TIME_FOUR_HOUR,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_4H.csv')
BuildCsv(TIME_ONE_HOUR,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_1H.csv')
BuildCsv(TIME_FIFTEEN_MINUTE,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_15M.csv')
BuildCsv(TIME_FIVE_MINUTE,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_5M.csv')

DataName = 'SOLUSDT'
BuildCsv(TIME_ONE_DAY,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_1D.csv')
BuildCsv(TIME_FOUR_HOUR,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_4H.csv')
BuildCsv(TIME_ONE_HOUR,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_1H.csv')
BuildCsv(TIME_FIFTEEN_MINUTE,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_15M.csv')
BuildCsv(TIME_FIVE_MINUTE,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_5M.csv')


DataName = 'XRPUSDT'
BuildCsv(TIME_ONE_DAY,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_1D.csv')
BuildCsv(TIME_FOUR_HOUR,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_4H.csv')
BuildCsv(TIME_ONE_HOUR,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_1H.csv')
BuildCsv(TIME_FIFTEEN_MINUTE,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_15M.csv')
BuildCsv(TIME_FIVE_MINUTE,DataName,'G:/CsvStorage/'+DataName+'/'+DataName+'_5M.csv')

'''


'''


   if(cutted.shape[0]!= dataframe.shape[0]):
        if dataframe.shape[0] >= cutted.shape[0]:
            value_difference = dataframe.shape[0] - cutted.shape[0]
            print(dataframe.shape[0] -dataframe_current.shape[0] )
            cutted = (dataframe[dataframe.shape[0] -dataframe_current.shape[0] + value - value_difference:])
            new_data = np.concatenate([cutted,dataframe_current], axis=1)
        else:
            value_difference = cutted.shape[0] -dataframe.shape[0]
            cutted = (dataframe_current[dataframe_current.shape[0] - dataframe.shape[0] + value - value_difference:])
            new_data = np.concatenate([dataframe, cutted], axis=1)
'''