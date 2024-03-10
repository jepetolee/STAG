import numpy as np
import pandas as pd

def normalize_data(data):
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    normalized_data = (data - mean) / std
    return normalized_data

def MovingAverageWithMaxAbsScaler(ClosePrices: pd.DataFrame, HyperParameter: int):

    MovingAverage = ClosePrices.rolling(window=HyperParameter).mean()
    MovingAverage = normalize_data(MovingAverage[HyperParameter:].values.reshape(1,-1))
    negative_twos_array = np.full(HyperParameter, -2).reshape(1,-1)
    MovingAverage = np.concatenate((negative_twos_array, MovingAverage),axis=1)
    return MovingAverage


def BollingerBandWithMaxAbsScaler(ClosePrices: pd.DataFrame, RollingValue: int, DevWeight: int = 2):

    BollingerBandMiddle = ClosePrices.rolling(window=RollingValue).mean()
    StandardDeviation = ClosePrices.rolling(window=RollingValue).std()
    negative_twos_array = np.full(RollingValue, -2).reshape(1,-1)

    BollingerBandUpper = BollingerBandMiddle + DevWeight * StandardDeviation
    BollingerBandLower = BollingerBandMiddle - DevWeight * StandardDeviation

    BollingerBandMiddle = normalize_data(BollingerBandMiddle[RollingValue:].values.reshape(1,-1))
    BollingerBandMiddle = np.concatenate((negative_twos_array, BollingerBandMiddle),axis=1)

    BollingerBandUpper = normalize_data(BollingerBandUpper[RollingValue:].values.reshape(1,-1))
    BollingerBandUpper = np.concatenate((negative_twos_array, BollingerBandUpper),axis=1)

    BollingerBandLower = normalize_data(BollingerBandLower[RollingValue:].values.reshape(1,-1))
    BollingerBandLower = np.concatenate((negative_twos_array, BollingerBandLower),axis=1)

    return np.concatenate((BollingerBandUpper, BollingerBandMiddle,BollingerBandLower),axis=0)


def RSIValue(ClosePrices: pd.DataFrame, HyperParameter: int):

    variance = ClosePrices - ClosePrices.shift(1)
    rise_width = variance.where(variance >= 0, 0)
    degrade_width = variance.where(variance < 0, 0).abs()
    AverageUp = rise_width.ewm(com=HyperParameter - 1, min_periods=HyperParameter, adjust=False).mean()
    AverageDown = degrade_width.ewm(com=HyperParameter - 1, min_periods=HyperParameter, adjust=False).mean()
    RSI = AverageUp / (AverageUp + AverageDown)
    arr = np.concatenate((np.full(HyperParameter-1,-2).reshape(1,-1), RSI[HyperParameter-1:].values.reshape(1,-1)),axis=1)
    fixed = np.nan_to_num(arr, nan=-2)
    return fixed


def KDJValue(HighPrices: pd.DataFrame, LowPrices: pd.DataFrame, ClosePrices: pd.DataFrame,
                        HyperParameter: int):

    K = np.full(len(ClosePrices.values),-2, dtype=float)
    D = np.full(len(ClosePrices.values),-2, dtype=float)
    J = np.full(len(ClosePrices.values),-2, dtype=float)

    for i in range(HyperParameter, len(LowPrices)):

        high = HighPrices.values[i - HyperParameter + 1:i + 1]
        low = LowPrices.values[i - HyperParameter + 1:i + 1]
        close = ClosePrices.values[i - HyperParameter + 1:i + 1]

        # Fast K 계산
        divider = np.max(high) - np.min(low)
        if divider ==0:
            divider =1

        rsv = (close[-1] - np.min(low)) / (divider)
        K[i] = rsv

        # Fast D 계산 (3일간의 이동평균)
        D[i] = np.mean(K[i - 2:i + 1])

        # Slow K, Slow D 계산
        J[i] = 3 * K[i] - 2 * D[i]

    return np.concatenate((K.reshape(1,-1), D.reshape(1,-1), J.reshape(1,-1)),axis=0)
