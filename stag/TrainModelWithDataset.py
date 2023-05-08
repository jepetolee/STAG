from Model import *
from DatasetBuilder import *
import torch
import math
import random
from Model import TradingA2C
from torchvision import transforms
from torch.distributions import Categorical
gamma = 0.75
weight = 0.84
tau = 0.97

def TrainWithDataset(crypto_name, load_model=False, adder=0, device='cuda'):
    model = TradingA2C().to(device)
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Grayscale()])
    if load_model is True:
        model.load_state_dict(torch.load('./models.pt'))
    testing_number_data = TakeCsvData('G:/CsvStorage/' + crypto_name + '/BTCUSDT_15M.csv')
    virtual_trader = RL_Agent(leverage=10, testmode=True)
    feature_list =list()

    GAE,count =0,0
    prekeepRemains = False
    PositionOptimizer  =  torch.optim.AdamW(list(model.feature_extract.parameters())+list(model.PiNet.parameters()), lr=1e-3, betas=(0.9, 0.999))
    KeepingOptimizer = torch.optim.AdamW(list(model.ValueNet.parameters())+list(model.DecidingNet.parameters()), lr=1e-3, betas=(0.9, 0.999))
    for number in range(len(testing_number_data) - adder - 19199):

        decided_number = number + adder
        url = 'G:/ImgDataStorage/' + crypto_name + '/COMBINED/' + str(decided_number + 1) + '.jpg'
        crypto_chart = PIL.Image.open(url)
        crypto_chart = trans(crypto_chart).float().to(device)
        feature_list.append(crypto_chart)
        count+=1

        if count >= 40:
            video = torch.stack(feature_list,dim=0).to(device).reshape(-1,40,3000,2400)
            close_price_in_csv_data = testing_number_data['ClosePrice'][19199 + decided_number]
            feature_list.pop(0)
            if virtual_trader.CheckActionSelectMode == CheckActionSelectModeTrue:
                action = model.Pi(video)
                PositionVideo = video.cpu()
                act = torch.argmax(action, dim=1).reshape(-1, 1)
                virtual_trader.PositionPrice = close_price_in_csv_data
                virtual_trader.check_position(act)
                delay_count = count+2
            elif count>delay_count and virtual_trader.CheckActionSelectMode == CheckActionSelectModeFalse:
                prekeepRemains = True
                keeping = model.Deciding(video)
                CheckKeep = torch.argmax(keeping, dim=1).reshape(-1, 1)
                virtual_trader.check_keeping(CheckKeep)

            virtual_trader.checking_bankrupt()
            virtual_trader.check_price_type(close_price_in_csv_data)

            if count > delay_count+1:
                reward = virtual_trader.get_reward()
                virtual_trader.PGAE = virtual_trader.PGAE * gamma * tau
                reward = torch.tensor(reward, dtype=torch.float32, device=device).reshape(-1, 1)
                if virtual_trader.CheckActionSelectMode == CheckActionSelectModeUNTRAINED:
                        virtual_trader.CheckActionSelectMode = CheckActionSelectModeTrue
                        rewardP = virtual_trader.ChoicingTensor.to(device)
                        GammaValueP = model.ValueP(video)
                        virtual_trader.PGAE = virtual_trader.PGAE+rewardP
                        GAE = 0
                        TargetVectorP = virtual_trader.PGAE + gamma * GammaValueP
                        SettingsValueP = model.ValueP(PositionVideo.to(device))
                        AdventageP = (TargetVectorP - SettingsValueP).detach() * tau
                        PiValue = model.Pi(video)
                        PiDist = Categorical(PiValue)

                        ActionProbability = Categorical(action.view(-1))
                        PolicyP = ActionProbability.log_prob(act)
                        RatioP = torch.exp(PiDist.log_prob(act) - PolicyP)
                        surrogated_loss1P = RatioP * AdventageP
                        surrogated_loss2P = torch.clamp(RatioP, 0.9, 1.1) * AdventageP
                        PositionLoss = - torch.min(surrogated_loss1P, surrogated_loss2P) \
                               + F.smooth_l1_loss(SettingsValueP, TargetVectorP.detach()) * weight

                        PositionOptimizer.zero_grad()
                        PositionLoss.backward(retain_graph=True)
                        PositionOptimizer.step()

                GAE = GAE * gamma * tau
                GAE = GAE +reward
                for i in range(3):
                    GammaValue = model.Value(video)
                    TargetVector = GAE + gamma*GammaValue
                    SettingsValue = model.Value(PositionVideo.to(device))
                    Adventage = (TargetVector - SettingsValue).detach()
                    DecidngValue = model.Deciding(video)
                    DecidngDist = Categorical(DecidngValue)

                    KeepingProbability = Categorical(pre_keeping.view(-1))
                    Policy = KeepingProbability.log_prob(CheckKeep)
                    Ratio = torch.exp(DecidngDist.log_prob(CheckKeep) - Policy)
                    surrogated_loss1 = Ratio * Adventage
                    surrogated_loss2 = torch.clamp(Ratio, 0.9, 1.1) * Adventage
                    loss = - torch.min(surrogated_loss1, surrogated_loss2) \
                               + F.smooth_l1_loss(SettingsValue, TargetVector.detach()) * weight
                    KeepingOptimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    KeepingOptimizer.step()
                GAE = TargetVector

                if virtual_trader.CurrentPosition ==0:
                    position = "NONE"
                elif virtual_trader.CurrentPosition ==1:
                    position = "Buy"
                elif virtual_trader.CurrentPosition ==2:
                    position = "Sell"

                if decided_number % 150 == 1:
                    print(reward.item(),
                            str(virtual_trader.PercentState) + "% Now ",
                            str(virtual_trader.UndefinedPercent) + "% Profit ",
                            position,
                            " Postion Price: "+ str(virtual_trader.PositionPrice),
                            "Now Price: " +   str(close_price_in_csv_data))

            if prekeepRemains:
                pre_keeping = keeping.detach()
                prekeepRemains =False

        if decided_number % 1000 == 1:
            torch.save(model.state_dict(),'./models.pt')


if __name__ == '__main__':
    import PIL

    TrainWithDataset('BTCUSDT', load_model=False, adder=0)
