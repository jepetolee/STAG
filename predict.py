from stag.DatasetBuilder import *
from stag.Model import TradingSupervisedModel,TradingSupervisedValueModel
import numpy as np
import torch

import requests


def predict(DataName,device = 'cuda'):
    BuildCsv(TIME_ONE_DAY, DataName, 'G:/CsvStorage/Trade/' + DataName + '_1D.csv',
             start_date='2023-06-01')
    BuildCsv(TIME_FOUR_HOUR, DataName, 'G:/CsvStorage/Trade/' + DataName + '_4H.csv',
             start_date='2023-08-01')
    BuildCsv(TIME_ONE_HOUR, DataName, 'G:/CsvStorage/Trade/' + DataName + '_1H.csv',
             start_date='2023-01-01')
    BuildCsv(TIME_FIFTEEN_MINUTE, DataName, 'G:/CsvStorage/Trade/' + DataName + '_15M.csv',
             start_date='2024-01-18')
    BuildCsv(TIME_FIVE_MINUTE, DataName, 'G:/CsvStorage/Trade/' + DataName + '_5M.csv',
             start_date='2024-01-20')

    BuildPredictData(DataName)

    model = TradingSupervisedModel().to(device).eval()
    model.load_state_dict(torch.load('./stag/Model-zoo/Trader.pt'))
    output = model(torch.from_numpy(np.load('./Predict.npy')).to(device).float().reshape(-1, 1, 270, 240))
    position =torch.argmax(output, dim=1)

    model = TradingSupervisedValueModel().to(device).eval()
    model.load_state_dict(torch.load('./stag/Model-zoo/TraderValue.pt'))
    value = model(torch.from_numpy(np.load('./Predict.npy')).to(device).float().reshape(-1, 1, 270, 240))

    return  position.item(), value.item()


