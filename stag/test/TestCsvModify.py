from stag.DatasetBuilder import *

URL = ''


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
        UpdateCsv('BTCUSDT', URL)
    finally:
        print("Updating Csv Doesn't Work.")
