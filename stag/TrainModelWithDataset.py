from Model import *
from DatasetBuilder import *
import torch
from Model import TradingA2C
from torchvision import transforms
from torch.distributions import Categorical
gamma = 0.75
weight = 0.84
tau = 0.97

def TrainWithDataset(crypto_name, load_model=False, adder=0, device='cuda'):
    model = TradingA2C().to(device)


    trans = transforms.Compose([transforms.ToTensor()])
    if load_model is True:
        model.loading_model('./models.pt')
    testing_number_data = TakeCsvData('G:/CsvStorage/' + crypto_name + '/BTCUSDT_15M.csv')
    virtual_trader = RL_Agent(leverage=10, testmode=True)
    feature_list =list()
    count=0

    PositionOptimizer  =  torch.optim.AdamW(list(model.feature_extract.parameters())+list(model.ValueNet.parameters())+list(model.PiNet.parameters()), lr=1e-3, betas=(0.9, 0.999))
    KeepingOptimizer = torch.optim.AdamW(list(model.feature_extract.parameters())+list(model.ValueNet.parameters())+list(model.DecidingNet.parameters()), lr=1e-3, betas=(0.9, 0.999))
    for number in range(len(testing_number_data) - adder - 19199):

        decided_number = number + adder
        url = 'G:/ImgDataStorage/' + crypto_name + '/COMBINED/' + str(decided_number + 1) + '.jpg'
        crypto_chart = PIL.Image.open(url)
        crypto_chart =  crypto_chart.resize((1500,1200))
        crypto_chart = trans(crypto_chart).float().to(device)
        feature_list.append(crypto_chart)
        count+=1

        if count >= 50:
            video = torch.stack(feature_list,dim=0).to(device).reshape(-1,50,3,1500,1200).permute(0,2,1,3,4)
            feature_list.pop(0)
            if virtual_trader.CheckActionSelectMode:
                action = model.Pi(video)
                act = torch.argmax(action, dim=1).reshape(-1, 1)
                virtual_trader.check_position(act)
            else:
                keeping = model.Deciding(video)
                CheckKeep = torch.argmax(keeping, dim=1).reshape(-1, 1)
                virtual_trader.check_keeping(CheckKeep)

            close_price_in_csv_data = testing_number_data['ClosePrice'][19199 + decided_number]
            virtual_trader.checking_bankrupt()
            virtual_trader.check_price_type(close_price_in_csv_data)

            if count >= 51:
                for i in range(3):
                    pre_keeping = keeping.detach()
                    if virtual_trader.CheckActionSelectMode:
                        dist = F.softmax(virtual_trader.ChoicingTensor)
                        PositionLoss = F.mse_loss(dist,action)
                        PositionOptimizer.zero_grad()
                        PositionLoss.backward(retain_graph=True)
                        PositionOptimizer.step()
                    else:
                        GammaValue = model.Value(pre_video)
                        TargetVector = reward + gamma * GammaValue
                        SettingsValue = model.Value(video)
                        Adventage = (TargetVector - SettingsValue).detach() * tau
                        DecingValue = model.Deciding(video)
                        DecingDist = Categorical(DecingValue)

                        KeepingProbability = Categorical(pre_keeping.view(-1))
                        Policy = KeepingProbability.log_prob(KeepingProbability.sample())

                        Ratio = torch.exp(DecingDist.log_prob(CheckKeep) - Policy)
                        surrogated_loss1 = Ratio * Adventage
                        surrogated_loss2 = torch.clamp(Ratio, 0.9, 1.1) * Adventage
                        loss = - torch.min(surrogated_loss1, surrogated_loss2).mean() \
                               + F.smooth_l1_loss(SettingsValue, TargetVector.detach()).mean() * weight
                        KeepingOptimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        KeepingOptimizer.step()

                if decided_number % 150 == 1:
                    print(action, reward, str(virtual_trader.PercentState) + "%",
                              str(virtual_trader.UndefinedPercent) + "%",
                              virtual_trader.CurrentPosition, virtual_trader.PositionPrice, close_price_in_csv_data)

            reward = virtual_trader.get_reward()
            reward = torch.tensor(reward,dtype=torch.float32,device=device).reshape(-1,1)
            pre_video = video.detach()
        if decided_number % 1000 == 1:
            torch.save(model.state_dict(),'./models.pt')


if __name__ == '__main__':
    import PIL

    TrainWithDataset('BTCUSDT', load_model=False, adder=0)
