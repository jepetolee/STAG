import torch
import torch.nn.functional as F
import torch.distributions as Dist
from torch import nn, jit


class DenseModel(jit.ScriptModule):
    def __init__(self, feature_size, output_shape , hidden_size, dist='normal'):
        super().__init__()
        # Models
        self.FullyConnected1 = nn.Linear(feature_size, hidden_size)
        self.FullyConnected2 = nn.Linear(hidden_size, hidden_size)
        self.FullyConnected3 = nn.Linear(hidden_size, output_shape)
        # Ex- variables to use
        self.OutputShape = output_shape
        self.DistributionOption = dist

    @jit.script_method
    def forward(self, features):
        AfterFC_Layer1 = F.elu(self.FullyConnected1(features))
        AfterFC_Layer2 = F.elu(self.FullyConnected2(AfterFC_Layer1))
        DistributionalInputs = F.elu(self.FullyConnected3(AfterFC_Layer2))
        reshaped_inputs = torch.reshape(DistributionalInputs, features.shape[:-1] + self.OutputShape)
        NormalizedOutput = Dist.Normal(reshaped_inputs, 1)
        return Dist.independent.Independent(NormalizedOutput, len(self.OutputShape))
