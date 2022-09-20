import torch
import torch.nn.functional as F
import torch.distributions
from torch import nn, jit

from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from Distribution import TanhBijector, SampleDist


class ActorModel(jit.ScriptModule):
    def __init__(self, belief_size, state_size, hidden_size, action_size,
                 min_std=1e-4, init_std=5, mean_scale=5):
        super().__init__()

        self.FullyConnected1 = nn.Linear(belief_size + state_size, hidden_size)
        self.FullyConnected2 = nn.Linear(hidden_size, hidden_size)
        self.FullyConnected3 = nn.Linear(hidden_size, hidden_size)
        self.FullyConnected4 = nn.Linear(hidden_size, hidden_size)
        self.FullyConnected5 = nn.Linear(hidden_size, 2 * action_size)

        self.MinStd = min_std
        self.InitStd = init_std
        self.MeanScale = mean_scale

    @jit.script_method
    def forward(self, belief, state):
        RawInitStd = torch.log(torch.exp(self.InitStd) - 1)
        x = torch.cat([belief, state], dim=1)
        AfterFC_Layer1 = F.elu(self.FullyConnected1(x))
        AfterFC_Layer2 = F.elu(self.FullyConnected2(AfterFC_Layer1))
        AfterFC_Layer3 = F.elu(self.FullyConnected3(AfterFC_Layer2))
        AfterFC_Layer4 = F.elu(self.FullyConnected4(AfterFC_Layer3))
        action = self.FullyConnected5(AfterFC_Layer4).squeeze(dim=1)

        MeanOfAction, StdDeviationOfAction = torch.chunk(action, 2, dim=1)
        MeanOfAction = self.MeanScale * torch.tanh(MeanOfAction / self.MeanScale)
        ActionStd = F.softplus(StdDeviationOfAction + RawInitStd) + self.MinStd
        return MeanOfAction, ActionStd

    def get_action(self, belief, state, Det=False):
        MeanOfAction, ActionStd = self.forward(belief, state)
        NormalizedAction = Normal(MeanOfAction, ActionStd)
        DistributedAction = TransformedDistribution(NormalizedAction, TanhBijector())
        OneDimensionedAction = torch.distributions.Independent(DistributedAction, 1)
        Distribution = SampleDist(OneDimensionedAction)
        if Det:
            return Distribution.mode()
        else:
            return Distribution.rsample()
