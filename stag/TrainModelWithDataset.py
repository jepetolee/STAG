from Model import *
import DatasetBuilder
from DatasetBuilder import *
from tqdm import trange
import PIL
import torch
from torch.distributions import Categorical
from Model import TradingModel
from torchvision import transforms


def TrainWithDataset(crypto_name, load_model=False, adder=0, device='cuda', train_steps_per_trade=10):
    model = TradingModel().to(device)
    trans = transforms.Compose([transforms.ToTensor()])
    if load_model is True:
        model.loading_model('./models.pt')
    testing_number_data = TakeCsvData('G:/CsvStorage/' + crypto_name + '/BTCUSDT_1H.csv')
    virtual_trader = RL_Agent(leverage=10, testmode=True)
    feature_list =list()
    count=0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999))
    for number in range(len(testing_number_data) - adder - 19000):

        decided_number = number + adder
        url = 'G:/ImgDataStorage/' + crypto_name + '/COMBINED/' + str(decided_number + 1) + '.jpg'
        crypto_chart = PIL.Image.open(url)
        crypto_chart = crypto_chart.resize((750,600))
        crypto_chart = trans(crypto_chart).float().to(device)
        feature_list.append(crypto_chart)
        count+=1
        if count >= 96:
            video = torch.stack(feature_list,dim=0).to(device)
            feature_list.pop(0)
            action = model(video.reshape(-1,3,750,600),device)
            close_price_in_csv_data = testing_number_data['ClosePrice'][LeastNumber2Build + 4560 + decided_number]
            virtual_trader.check_position(action)
            virtual_trader.check_price_type(close_price_in_csv_data)
            reward = virtual_trader.get_reward()
            loss = torch.sum(action,dim=1).reshape(-1)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            print(action, reward, str(virtual_trader.PercentState) + "%", str(virtual_trader.UndefinedPercent) + "%",
                  virtual_trader.CurrentPosition, virtual_trader.PositionPrice, close_price_in_csv_data)

        '''if decided_number % 1000 == 1:
            model.saving_model('./models.pt')'''


if __name__ == '__main__':
    import PIL

    TrainWithDataset('BTCUSDT', load_model=False, adder=0)
