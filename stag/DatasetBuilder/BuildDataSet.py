from stag.DatasetBuilder.MakeCsv2Img import *
'''
ex> 'C:/Users/Adminstrator/DataStorage/BTC_USDT/'
'''
URL_Slash ='/'
def detailed_dataset_root(crypto_name):
    return IMGFileMainRoot + URL_Slash + crypto_name  +URL_Slash+'15min_'+crypto_name+'.csv'


