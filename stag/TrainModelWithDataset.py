from Model import *
import DatasetBuilder
from DatasetBuilder import *
from tqdm import trange


def TrainWithDataset(crypto_name, device='cpu', train_steps_per_trade=10):
    model = Dreamer(device=device, train_steps=train_steps_per_trade)
    image_data = DatasetBuilder.Dataset(crypto_name)
    testing_number_data = TakeCsvData(detailed_dataset_root(crypto_name, '15m'))
    virtual_trader = RL_Agent(leverage=20)

    reward_stack = list()
    chart_stack = list()

    for time_steps in trange(image_data.size):
        crypto_chart = image_data.call_image_tensor(time_steps)
        action, action_distribution, value_prediction, reward_prediction, posterior_state = model.RunModel(crypto_chart)
        close_price_in_csv_data = testing_number_data['3'][LeastNumber2Build + time_steps]

        virtual_trader.check_position(action)
        virtual_trader.check_price_type(close_price_in_csv_data)
        reward, DoesDone = virtual_trader.get_reward()

        if DoesDone is True:
            reward_tensor = torch.tensor(reward_stack).reshape(-1,1)
            chart_tensor = torch.tensor(chart_stack).reshape(-1,1,1,1)
            model.train_data(reward_tensor)

            reward_stack.clear()
            chart_stack.clear()
            del reward_tensor
            del chart_tensor


