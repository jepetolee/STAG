from Model import *
import DatasetBuilder
from DatasetBuilder import *
from tqdm import trange
import PIL
import torch
from torch.distributions import Categorical

from torchvision import transforms
def TrainWithDataset(crypto_name, load_model = False,device='cuda', train_steps_per_trade=10):
    model = Dreamer(device=device, train_steps=train_steps_per_trade)
    trans = transforms.Compose([transforms.PILToTensor(),
                                transforms.Resize(size=(666, 2475))])
    if load_model is True:
        model.loading_model('./models.pt')
    testing_number_data = TakeCsvData(detailed_dataset_root(crypto_name))
    virtual_trader = RL_Agent(leverage=20)
    for time_steps in trange(len(testing_number_data) -3200+1):
        url = 'G:/'+crypto_name+'/COMBINED/'+str(time_steps+1)+'.png'
        crypto_chart = PIL.Image.open(url)
        crypto_chart = trans(crypto_chart).float().to(device).reshape(-1,3,666,2475)
        action, action_distribution, value_prediction, reward_prediction, posterior_state = model.RunModel(crypto_chart)
        close_price_in_csv_data = testing_number_data['ClosePrice'][LeastNumber2Build +3000+ time_steps]
        virtual_trader.check_position(action)
        virtual_trader.check_price_type(close_price_in_csv_data)
        reward = virtual_trader.get_reward()
        reward =  torch.tensor(reward).float().to(device)
        model.OptimizeModel(crypto_chart,action,reward)
        model.saving_model('./models.pt')


if __name__ == '__main__' :
    import PIL

    TrainWithDataset('BTCUSDT')



