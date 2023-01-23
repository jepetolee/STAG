from Model import *
import DatasetBuilder
from DatasetBuilder import *
from tqdm import trange
import PIL
import torch
def TrainWithDataset(crypto_name, device='cuda', train_steps_per_trade=10):
    model = Dreamer(device=device, train_steps=train_steps_per_trade)
    trans = transforms.Compose([transforms.PILToTensor(),
                                transforms.Resize(size=(666, 2475))])
    testing_number_data = TakeCsvData(detailed_dataset_root(crypto_name))
    virtual_trader = RL_Agent(leverage=20)

    for time_steps in trange(testing_number_data.size[0] -3200):
        url  = 'G:/'+crypto_name+'/COMBINED'+str(time_steps+1)+'.png'
        crypto_chart = PIL.Image.open(url)
        crypto_chart = trans(crypto_chart).float().to(device).reshape(-1,3,666,2475)
        action, action_distribution, value_prediction, reward_prediction, posterior_state = model.RunModel(crypto_chart)
        close_price_in_csv_data = testing_number_data['3'][LeastNumber2Build +3000+ time_steps]

        virtual_trader.check_position(action)
        virtual_trader.check_price_type(close_price_in_csv_data)
        reward = virtual_trader.get_reward()
        reward =  torch.tensor(reward,torch.float32,device)
        model.OptimizeModel(crypto_chart,action,reward)


if __name__ == '__main__' :
    import PIL
    from torchvision import transforms

    Img = PIL.Image.open('G:/BTCUSDT/COMBINED/1.png')
    trans = transforms.Compose([transforms.PILToTensor(),
                                transforms.Resize(size=(666, 2475))])
    img = trans(Img).float().to('cuda').reshape(-1,3,666,2475)

    model = Dreamer('cuda', 10)

    action, action_distribution, value, reward, state = model.RunModel(img)

    model.OptimizeModel(img,action,reward.sample())
    print(action.shape)
    print(action_distribution.sample())
    print(value)
    print(reward)
    print(state)


