from Model import *
import DatasetBuilder
import torch
from DatasetBuilder import *
from tqdm import trange


def Train(crypto_name, device='cpu', train_steps_per_trade=10):
    model = Dreamer(device=device, train_steps=train_steps_per_trade)
    image_data = DatasetBuilder.Dataset(crypto_name)
    testing_number_data = TakeCsvData(detailed_dataset_root(crypto_name, '15m'))
    virtual_trader = RL_Agent(leverage=20)

    for time_steps in trange(image_data.size):
        crypto_chart = image_data.call_image_tensor(time_steps)
        average_price = testing_number_data['3'][LeastNumber2Build + time_steps]
        action, action_distribution, value_prediction, reward_prediction, posterior_state = model.RunModel(crypto_chart)
        