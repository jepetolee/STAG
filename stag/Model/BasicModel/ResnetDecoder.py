
import torch
from torch import nn, jit
from torch.nn import functional as F
import numpy as np
import torch.distributions as dist
from torchvision import transforms


class BasicDecoderBlock(nn.Module):
    def __init__(self, inputs, depth):
        super().__init__()
        self.depth = depth
        self.residual1 = nn.Sequential(nn.ConvTranspose2d(inputs, depth, 1, stride=2, bias=False), nn.BatchNorm2d(3), nn.ELU())
        self.conv1 = nn.Sequential(nn.BatchNorm2d(inputs), nn.ELU(), nn.ConvTranspose2d(inputs, depth, 3, stride=2))
        self.conv2 = nn.Sequential(nn.BatchNorm2d(inputs), nn.ELU(), nn.ConvTranspose2d(depth, depth, 3, stride=2))

    def forward(self, image):
        skip = image
        if image.shape[-1] != self.depth:
            skip = self.residual1(image)
        x = self.conv1(image)
        x = self.conv2(x)
        return skip + 0.1 * x

class ObservationDecoder(nn.Module):

    def __init__(self, stochastic_size, deterministic_size, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.shape =(3,1000,3750)
        self.FullyConnected = nn.Linear(stochastic_size + deterministic_size, embedding_size)
        self.Conv1 = BasicDecoderBlock(256, 128)
        self.Conv2 = BasicDecoderBlock(128, 64)
        self.Conv3 = BasicDecoderBlock(64, 32)
        self.Conv4 = BasicDecoderBlock(32, 3)

    def forward(self, x):
        batch_shape = x.shape[:-1]
        embed_size = x.shape[-1]
        squeezed_size = np.prod(batch_shape).item()
        x = x.reshape(squeezed_size, embed_size)
        AfterLinear = F.elu(self.FullyConnected(x))

        AfterLinear = torch.reshape(AfterLinear,(-1, 256, 39, 152))
        AfterConv1 = self.Conv1(AfterLinear)
        AfterConv2 = self.Conv2(AfterConv1)
        AfterConv3 = self.Conv3(AfterConv2)

        observation =self.Conv4(AfterConv3)
        trans = transforms.Compose([transforms.Resize(size=(666, 2475))])

        observation = trans(observation)

        obs_dist = dist.Independent(dist.Normal(observation, 1), len(self.shape))
        return obs_dist