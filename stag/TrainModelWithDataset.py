from Model import *
from DatasetBuilder import *
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import math
import random

gamma = 0.75
weight = 0.84
tau = 0.97

EPS_START = 0.95  # 학습 시작시 에이전트가 무작위로 행동할 확률
EPS_END = 0.04 # 학습 막바지에 에이전트가 무작위로 행동할 확률
EPS_DECAY = 800000  # 학습 진행시 에이전트가 무작위로 행동할 확률을 감소시키는 값
EPS_DECAY2 = 900000
GAMMA = 0.8


def TrainWithDataset(crypto_name, load_model=False, adder=0, device='cuda', lack=0,lack2=0):
    model = TradingConvLSTMA2C().to(device)
    if load_model is True:
        model.load_state_dict(torch.load('./killermodels.pt'))
    testing_number_data = TakeCsvData('G:/CsvStorage/' + crypto_name + '/' + crypto_name + '_1M.csv')
    virtual_trader = RL_Agent(leverage=20, testmode=True ,limitymode=True)
    feature_list = list()
    count,Bankrupt = 0,0
    prekeepRemains = False
    PositionOptimizer = torch.optim.AdamW(
        list(model.conv.parameters()) + list(model.feature_extract.parameters()) + list(
            model.PiNet.parameters()) + list(model.ValuePNet.parameters())+list(model.lstmPi.parameters())+list(model.lstmPiValue.parameters()), lr=1e-3, betas=(0.9, 0.999))
    KeepingOptimizer = torch.optim.AdamW(
        list(model.conv.parameters()) + list(model.feature_extract.parameters()) + list(
            model.DecidingNet.parameters()) + list(model.ValueNet.parameters())+list(model.lstmDecide.parameters())+list(model.lstmDecideValue.parameters()), lr=1e-3, betas=(0.9, 0.999))

    model.reset_lstm()
    pbar = trange(len(testing_number_data) - adder - LeastNumber2Build - 287799)
    for number in pbar:
        decided_number = number + adder
        url = 'G:/ParquetDataStorage/' + crypto_name + '/' + str(decided_number + 1) + '.parquet'
        data_columns = ['ClosePrice','HighPrice','LowPrice','OpenPrice','TwentyAvg', 'FiftyAvg', 'HundredAvg', 'Volume', 'BollingerBandUpper','BollingerBandLower', 'RSI', 'PercentK', 'PercentD', 'PercentJ',
                        'ClosePrice5','HighPrice5','LowPrice5','OpenPrice5', 'TwentyAvg5', 'FiftyAvg5', 'HundredAvg5', 'Volume5', 'BollingerBandUpper5','BollingerBandLower5', 'RSI5', 'PercentK5', 'PercentD5', 'PercentJ5',
                        'ClosePrice15', 'HighPrice15', 'LowPrice15', 'OpenPrice15', 'TwentyAvg15', 'FiftyAvg15', 'HundredAvg15','Volume15', 'BollingerBandUpper15',
                        'BollingerBandLower15', 'RSI15', 'PercentK15', 'PercentD15', 'PercentJ15',
                        'ClosePrice1h','HighPrice1h','LowPrice1h','OpenPrice1h', 'TwentyAvg1h', 'FiftyAvg1h', 'HundredAvg1h', 'Volume1h', 'BollingerBandUpper1h',
                        'BollingerBandLower1h', 'RSI1h', 'PercentK1h', 'PercentD1h', 'PercentJ1h',
                        'ClosePrice4h','HighPrice4h','LowPrice4h','OpenPrice4h', 'TwentyAvg4h', 'FiftyAvg4h', 'HundredAvg4h', 'Volume4h', 'BollingerBandUpper4h',
                        'BollingerBandLower4h', 'RSI4h', 'PercentK4h', 'PercentD4h', 'PercentJ4h',
                        'ClosePrice1d','HighPrice1d','LowPrice1d','OpenPrice1d', 'TwentyAvg1d', 'FiftyAvg1d', 'HundredAvg1d', 'Volume1d', 'BollingerBandUpper1d',
                        'BollingerBandLower1d', 'RSI1d', 'PercentK1d', 'PercentD1d', 'PercentJ1d']
        crypto_chart = pd.read_parquet(url, columns=data_columns)
        crypto_chart = crypto_chart.to_numpy()
        crypto_chart = torch.from_numpy(crypto_chart).float()
        feature_list.append(crypto_chart)
        count += 1

        if count >= 80:
            video = torch.stack(feature_list, dim=0).to(device)
            close_price_in_csv_data = testing_number_data['ClosePrice'][decided_number + 287799 + LeastNumber2Build]
            feature_list.pop(0)
            if virtual_trader.CheckActionSelectMode == CheckActionSelectModeTrue:
                action = model.Pi(video)

                PositionVideo = video

                eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * (count + lack) / EPS_DECAY)

                if random.random() > eps_threshold:
                    act = torch.argmax(action, dim=1).reshape(-1, 1)
                else:
                    act = torch.LongTensor([[random.randrange(3)]]).reshape(-1, 1).to(device)
                virtual_trader.PositionPrice = close_price_in_csv_data
                virtual_trader.check_position(act)
                delay_count = count + 2
            elif count > delay_count and virtual_trader.CheckActionSelectMode == CheckActionSelectModeFalse:
                prekeepRemains = True

                keeping = model.Deciding(video)
                eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * (count + lack2) / EPS_DECAY2)

                if random.random() > eps_threshold:
                    CheckKeep = torch.argmax(keeping, dim=1).reshape(-1, 1)
                else:
                    rand =random.randint(1,350)
                    if rand == 50:
                        CheckKeep = torch.LongTensor([[0]]).reshape(-1, 1).to(device)
                    else:
                        CheckKeep = torch.LongTensor([[1]]).reshape(-1, 1).to(device)

                virtual_trader.check_keeping(CheckKeep)

            virtual_trader.checking_bankrupt()
            virtual_trader.check_price_type(close_price_in_csv_data)

            if count > delay_count + 1:
                reward, DoesDone = virtual_trader.get_reward()
                if DoesDone:
                    Bankrupt+=1
                    model.reset_lstm()
                reward = torch.tensor(reward, dtype=torch.float32, device=device).reshape(-1, 1)
                if virtual_trader.CheckActionSelectMode == CheckActionSelectModeUNTRAINED:
                    virtual_trader.CheckActionSelectMode = CheckActionSelectModeTrue
                    rewardP = virtual_trader.ChoicingTensor.to(device)
                    reward = virtual_trader.ChoicingTensor.to(device)
                    GammaValueP = model.ValueP(video)

                    TargetVectorP = rewardP + gamma * GammaValueP
                    SettingsValueP = model.ValueP(PositionVideo)
                    AdventageP = (TargetVectorP - SettingsValueP).detach() * tau
                    PiValue = model.Pi(video)
                    PiDist = Categorical(PiValue)

                    ActionProbability = Categorical(action.view(-1))
                    PolicyP = ActionProbability.log_prob(act)
                    RatioP = torch.exp(PiDist.log_prob(act) - PolicyP)
                    surrogated_loss1P = RatioP * AdventageP
                    surrogated_loss2P = torch.clamp(RatioP, 0.9, 1.1) * AdventageP
                    PositionLoss = - torch.min(surrogated_loss1P, surrogated_loss2P) + F.smooth_l1_loss(SettingsValueP, rewardP.detach()) * weight

                    PositionOptimizer.zero_grad()
                    PositionLoss.backward(retain_graph=True)
                    PositionOptimizer.step()

                GammaValue = model.Value(video)
                GAE = reward + gamma * GammaValue
                SettingsValue = model.Value(PositionVideo)
                Adventage = (GAE - SettingsValue).detach()

                DecidngValue = model.Deciding(video)
                DecidngDist = Categorical(DecidngValue)

                KeepingProbability = Categorical(pre_keeping.view(-1))
                Policy = KeepingProbability.log_prob(CheckKeep)
                Ratio = torch.exp(DecidngDist.log_prob(CheckKeep) - Policy)
                surrogated_loss1 = Ratio * Adventage
                surrogated_loss2 = torch.clamp(Ratio, 0.9, 1.1) * Adventage
                loss = - torch.min(surrogated_loss1, surrogated_loss2) + F.smooth_l1_loss(SettingsValue, reward.detach()) * weight
                KeepingOptimizer.zero_grad()
                loss.backward(retain_graph=True)
                KeepingOptimizer.step()
                del loss, surrogated_loss2, surrogated_loss1, Ratio, Policy, KeepingProbability, video

                if virtual_trader.CurrentPosition == 0:
                    position = "NONE"
                elif virtual_trader.CurrentPosition == 1:
                    position = "Buy"
                elif virtual_trader.CurrentPosition == 2:
                    position = "Sell"

                pbar.set_description(f"{virtual_trader.ChoicingTensor.item()}, "
                                     f"{str(virtual_trader.PercentState)}  % Now,"
                                     f"{str(virtual_trader.UndefinedPercent)}  % Profit,"
                                     f" Bankrupt Count:  {str(Bankrupt)}")

            if prekeepRemains:
                pre_keeping = keeping.detach()
                prekeepRemains = False

        if number % 1000 == 1:
            torch.save(model.state_dict(), './killermodels.pt')


if __name__ == '__main__':
    TrainStep = 4
    TrainWithDataset('BTCUSDT', load_model=False, adder=0)
    for i in range(TrainStep):
        TrainWithDataset('BTCUSDT', load_model=True, adder=150000, lack=300000)
"""
 {position}, "
                                     f" Postion Price:  {str(virtual_trader.PositionPrice)},"
                                     f" Now Price:  {str(close_price_in_csv_data)}"
"""