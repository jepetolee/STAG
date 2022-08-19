START_DATE = "2018-01-01"
TIME_ONE_DAY = '1d'
TIME_ONE_HOUR = '1h'
TIME_FOUR_HOUR = '4h'
TIME_FIFTEEN_MINUTE = '15m'
TIME_FIVE_MINUTE = '5m'

import pandas as pd
from binance.client import Client


def BuildCsv(time_interval, crypto_symbol, URL):
    client = Client(api_key='', api_secret='')
    k_lines = client.futures_historical_klines(symbol=crypto_symbol, interval=time_interval,
                                               start_str=START_DATE, limit=1000)
    dataframe = pd.DataFrame(k_lines)
    dataframe.to_csv(URL)
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

        csv_data = csv_data[['1', '2', '3', '4', '5']]
        csv_data.rename(columns={"1": "OpenPrice", "2": "HighPrice", "3": "LowPrice",
                                 "4": "ClosePrice", "5": "Volume"})
        csv_data = csv_data.set_index(readable_time_data)

        return csv_data

    except FileNotFoundError:
        print(f"Csv file was not found in {URL}")
        return
