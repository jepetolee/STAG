from Model import *
import DatasetBuilder
from DatasetBuilder import *
import random
import torch
from torch.distributions import Categorical
from Model import TradingA2C
from torchvision import transforms

gamma = 0.95
EPS_START = 0.95  # 학습 시작시 에이전트가 무작위로 행동할 확률
EPS_END = 0.05   # 학습 막바지에 에이전트가 무작위로 행동할 확률
EPS_DECAY = 1000
def TrainWithDataset(crypto_name, load_model=False, adder=0, device='cuda', train_steps_per_trade=10):
    model = TradingA2C().to(device)
    trans = transforms.Compose([transforms.ToTensor()])
    if load_model is True:
        model.loading_model('./models.pt')
    testing_number_data = TakeCsvData('G:/CsvStorage/' + crypto_name + '/BTCUSDT_1H.csv')
    virtual_trader = RL_Agent(leverage=10, testmode=True)
    feature_list =list()
    critertion = nn.SmoothL1Loss()
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

            steps_done = count - 96
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
            if random.random() > eps_threshold:
                act = torch.argmax(action, dim=1).reshape(-1, 1)
            else:
                act = torch.tensor([random.randint(0,2)],device=device).reshape(-1, 1)
            print(act)
            next_Q  = action.gather(1,act)

            close_price_in_csv_data = testing_number_data['ClosePrice'][19199 + decided_number]
            virtual_trader.check_position(act)
            virtual_trader.check_price_type(close_price_in_csv_data)
            if count >= 97:
                expected_Q = reward+gamma*next_Q
                loss = critertion(current_Q, expected_Q)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()


            reward = virtual_trader.get_reward()
            reward = torch.tensor(reward,dtype=torch.float32,device=device).reshape(-1,1)


            print(action, reward, str(virtual_trader.PercentState) + "%", str(virtual_trader.UndefinedPercent) + "%",
                  virtual_trader.CurrentPosition, virtual_trader.PositionPrice, close_price_in_csv_data)
            current_Q = next_Q

        if decided_number % 1000 == 1:
            torch.save(model.state_dict(),'./models.pt')


if __name__ == '__main__':
    import PIL

    TrainWithDataset('BTCUSDT', load_model=False, adder=0)
