from Model import *
import torch
from utils import *

model = TradingModel().cpu()

model.load_state_dict(torch.load('./BaseModel.pt'))

DrawingParameters(model, 'BTCUSDT')

for name, param in model.named_parameters():
    print(name)
    print(param)
    print(param.shape)
    print(torch.any(torch.isnan(param)))
