from Model import *
import DatasetBuilder
import torch
from tqdm import trange


def Train(crypto_name, device='cpu', train_steps_per_trade=10):
    model = Dreamer(device=device, train_steps=train_steps_per_trade)
    image_data = DatasetBuilder.Dataset(crypto_name)

    for time_steps in trange(image_data.size):
        crypto_chart = image_data.call_image_tensor(time_steps)
        action, action_distribution, value_prediction, reward_prediction, posterior_state = model.RunModel(crypto_chart)
