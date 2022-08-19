from stag.DatasetBuilder import *

URL = 'C:/Users/Administrator/Documents/GitHub/STAG/stag/test/BTCUSDT-1.csv'


def test():
    try:
        BuildCsv(TIME_ONE_HOUR, 'BTCUSDT', URL)
    finally:
        print("Building Csv Doesn't Work.")

    try:
        TakeCsvData(URL)
    finally:
        print("Taking Csv Data Doesn't Work.")

    try:
        UpdateCsv(TIME_ONE_HOUR,'BTCUSDT', URL)
    finally:
        print("Updating Csv Doesn't Work.")
