from Model import *
import DatasetBuilder
from DatasetBuilder import *
from tqdm import trange
import PIL
import torch
from torch.distributions import Categorical

from torchvision import transforms


def TrainWithDataset(crypto_name, load_model=False, adder=0, device='cuda', train_steps_per_trade=10):
    model = Dreamer(device=device, train_steps=train_steps_per_trade)
    trans = transforms.Compose([transforms.PILToTensor(),
                                transforms.Resize(size=(222, 825)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if load_model is True:
        model.loading_model('./models.pt')
    testing_number_data = TakeCsvData('G:/CsvStorage/' + crypto_name + '/BTCUSDT_1H.csv')
    virtual_trader = RL_Agent(leverage=10, testmode=True)

    action_list, reward_list, chart_list = list(), list(), list()

    for number in range(len(testing_number_data) - adder - 4760):
        decided_number = number + adder
        url = 'G:/ImgDataStorage/' + crypto_name + '/COMBINED/' + str(decided_number + 1) + '.jpg'
        crypto_chart = PIL.Image.open(url)
        crypto_chart = trans(crypto_chart).float().to(device)
        chart_list.append(crypto_chart)
        action, action_distribution, value_prediction, reward_prediction, posterior_state = model.RunModel(
            crypto_chart.reshape(-1, 3, 222, 825))
        close_price_in_csv_data = testing_number_data['ClosePrice'][LeastNumber2Build + 4560 + decided_number]
        virtual_trader.check_position(action)
        action_list.append(action)
        virtual_trader.check_price_type(close_price_in_csv_data)
        reward = virtual_trader.get_reward()
        print(action, reward, str(virtual_trader.PercentState) + "%", str(virtual_trader.UndefinedPercent) + "%",
              virtual_trader.CurrentPosition, virtual_trader.PositionPrice, close_price_in_csv_data)
        reward_list.append(reward)

        if decided_number % 10 == 0 and decided_number != 0:
            crypto_chart = torch.stack(chart_list, dim=0)
            action = torch.stack(action_list, dim=0)
            reward = torch.tensor(reward_list).float().to(device).reshape(-1, 1)
            model.OptimizeModel(crypto_chart, action, reward)
            action_list.clear()
            reward_list.clear()
            chart_list.clear()
        if decided_number % 1000 == 1:
            model.saving_model('./models.pt')


if __name__ == '__main__':
    import PIL

    TrainWithDataset('BTCUSDT', load_model=False, adder=0)
