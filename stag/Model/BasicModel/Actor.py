import torch
import torch.nn.functional as F
import torch.distributions
from torch import nn, jit
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from stag.Model.BasicModel.Distribution import SampleDist
from torch.distributions import TanhTransform


class ActorModel(nn.Module):
    def __init__(self, state_size, hidden_size, action_size,
                 min_std=1e-4, init_std=5, mean_scale=5):
        super().__init__()

        self.FullyConnected1 = nn.Linear(state_size, hidden_size)
        self.FullyConnected2 = nn.Linear(hidden_size, hidden_size)
        self.FullyConnected3 = nn.Linear(hidden_size, hidden_size)
        self.FullyConnected4 = nn.Linear(hidden_size, hidden_size)
        self.FullyConnected5 = nn.Linear(hidden_size,  action_size)#if tanh needto make *2

        self.MinStd = min_std
        self.InitStd = init_std
        self.MeanScale = mean_scale

    def forward(self, state, Det='one_hot'):
        RawInitStd = np.log(np.exp(self.InitStd) - 1)

        AfterFC_Layer1 = F.elu(self.FullyConnected1(state))
        AfterFC_Layer2 = F.elu(self.FullyConnected2(AfterFC_Layer1))
        AfterFC_Layer3 = F.elu(self.FullyConnected3(AfterFC_Layer2))
        AfterFC_Layer4 = F.elu(self.FullyConnected4(AfterFC_Layer3))
        action = self.FullyConnected5(AfterFC_Layer4).squeeze(dim=1)
        if Det == 'tanh_normal':
            MeanOfAction, StdDeviationOfAction = torch.chunk(action, 2, dim=1)
            MeanOfAction = self.MeanScale * torch.tanh(MeanOfAction / self.MeanScale)
            ActionStd = F.softplus(StdDeviationOfAction + RawInitStd) + self.MinStd
            NormalizedAction = Normal(MeanOfAction, ActionStd)
            DistributedAction = TransformedDistribution(NormalizedAction, TanhTransform())
            OneDimensionedAction = torch.distributions.Independent(DistributedAction, 1)
            return SampleDist(OneDimensionedAction)
        elif Det == 'one_hot':
            return torch.distributions.OneHotCategorical(logits=action)
        elif Det == 'relaxed_one_hot':
            return torch.distributions.RelaxedOneHotCategorical(0.1, logits=action)
