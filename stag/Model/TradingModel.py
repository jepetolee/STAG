import flask

from .BasicModel.Moe import MixedActor
from .BasicModel.efficientNet import get_efficientnet_v2
#from .BasicModel.DCGN import *
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
import gc
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import math
from .BasicModel.MoeLSTM import ResLSTMLayer
from .BasicModel.Conformer.encoder import ConformerEncoder
import torch


class GatingNetwork(nn.Module):
    def __init__(self, input_size, num_experts,feature=1024):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, feature)
        self.fc2 = nn.Linear(feature, num_experts)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


class ExpertNetwork(nn.Module):
    def __init__(self, input_size, output_size,feature=1024):
        super(ExpertNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, feature)
        self.fc2 = nn.Linear(feature, output_size)

    def forward(self, x):
        x =  F.gelu(self.fc1(F.dropout(x,p=0.1)))
        x = self.fc2(x)
        return x


class MixtureOfExperts(nn.Module):
    def __init__(self, input_size, output_size, num_experts,feature=512):
        super(MixtureOfExperts, self).__init__()
        self.gating_network = GatingNetwork(input_size, num_experts,feature)
        self.expert_networks = nn.ModuleList([ExpertNetwork(input_size, output_size,feature) for _ in range(num_experts)])

    def forward(self, x):
        gate_weights = self.gating_network(x)
        expert_outputs = [expert(x) for expert in self.expert_networks]

        # Combine expert outputs based on gate weights
        output = torch.stack(expert_outputs, dim=-1)  # Stack expert outputs along the last dimension
        output = torch.sum(output * gate_weights.unsqueeze(-2), dim=-1)  # Weighted sum

        return output



class TradingModel(nn.Module):
    def __init__(self):
        super(TradingModel, self).__init__()


        self.SFT = TradingSupervisedModel()
        self.SFT.load_state_dict(torch.load('./pretrained.pt'))
        self.SFT.eval()

        self.distribution = torch.distributions.Categorical
        self.memory = None
        self.MemoryInterpreter = nn.LSTM(input_size=1280,num_layers=4,hidden_size=1280)

        self.Decisioner = nn.Sequential(nn.Linear(1280,32),
                                         nn.GELU())
        self.DecisionA = MixtureOfExperts(32,2,8,feature=16)
        self.DecisionV = MixtureOfExperts(32,1,8,feature=16)

    def PositionAction(self, input,memory_mode=False):
        self.SFT.eval()
        x = self.SFT.conformer(input)

        H, C = self.memory[0].clone(), self.memory[1].clone()
        _, memory = self.MemoryInterpreter(x.reshape(-1, 1280), (H,C))

        if memory_mode:
            self.memory = memory

        A = F.softmax(self.SFT.Classifier(x),dim=1)

        return A

    def DecideAction(self, input, memory_mode=False):
        self.SFT.eval()
        x = self.SFT.conformer(input)

        H, C = self.memory[0].clone(), self.memory[1].clone()
        x, memory = self.MemoryInterpreter(x.reshape(-1, 1280), (H, C))
        if memory_mode:
            self.memory = memory
        x = self.Decisioner(x)
        A = F.softmax(self.DecisionA(x), dim=1)
        value = self.DecisionV(x)
        return A ,value
    def SEP(self, input):
        self.SFT.eval()
        x = self.SFT.conformer(input)
        return x
    def SEP_TRAIN(self,input,memory_mode=False):
        H, C = self.memory[0].clone(), self.memory[1].clone()
        x, memory = self.MemoryInterpreter(input.reshape(-1, 1280), (H, C))
        if memory_mode:
            self.memory = memory
        x = self.Decisioner(x)
        A = F.softmax(self.DecisionA(x), dim=1)
        Auxilary = self.DecisionAuxilary(x)
        value = self.DecisionV(x)
        return A ,Auxilary,value

    def Positionpretrain(self, input):
        return self.SFT(input)
    def BuildMemory(self):
        del self.memory
        gc.collect()
        torch.cuda.empty_cache()
        self.memory= (torch.zeros(4,1280).cuda().detach(), torch.zeros(4,1280).cuda().detach())
        return

    def DeleteMemory(self):
        del self.memory
        gc.collect()
        torch.cuda.empty_cache()
        return


class TradingSupervisedModel(nn.Module):
    def __init__(self):
        super(TradingSupervisedModel, self).__init__()

        self.conformer = get_efficientnet_v2()
        self.Classifier = nn.Sequential(nn.Linear(1280,32),
                                         nn.BatchNorm1d(32),
                                         nn.GELU(),
                                         nn.Linear( 32,3))
    def forward(self, input):

        x = self.conformer(input)
        return F.softmax(self.Classifier(x),dim=1)












class TradingSupervisedValueModel(nn.Module):
    def __init__(self):
            super(TradingSupervisedValueModel, self).__init__()

            self.conformer = get_efficientnet_v2()
            self.Classifier = nn.Sequential(nn.Linear(1280, 32),
                                            nn.BatchNorm1d(32),
                                            nn.GELU(),
                                            nn.Linear(32, 1))

    def forward(self, input):
            x = self.conformer(input)
            return self.Classifier(x)

