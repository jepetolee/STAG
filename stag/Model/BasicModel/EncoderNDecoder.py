import torch
from torch import nn, jit
from torch.nn import functional as F
import numpy as np
import torch.distributions as dist
from torchvision import transforms


class ObservationEncoder(nn.Module):
    def __init__(self,stride=2,shape=(3,1000,3750)):
        super().__init__()
        self.Conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.Conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.Conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.Conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.shape = shape
        self.depth = 32
        self.stride = stride
        self.FCLayer = nn.Identity()

    def forward(self, observation):
        Hidden1 = F.elu(self.Conv1(observation))
        Hidden2 = F.elu(self.Conv2(Hidden1))
        Hidden3 = F.elu(self.Conv3(Hidden2))
        Hidden4 = F.elu(self.Conv4(Hidden3))
        Output = self.FCLayer(Hidden4)
        Output = torch.reshape(Output,(Output.shape[0],-1))
        return Output

    @staticmethod
    def ShapeAfterConv(h_in, padding, kernel_size, stride):
        ShapeAfterCNN = list()
        for x in h_in:
            ShapeAfterCNN.append(int((x + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.))
        return tuple(ShapeAfterCNN)

    def embed_size(self):
        conv1_shape = self.ShapeAfterConv(self.shape[1:], 0, 4, self.stride)
        conv2_shape = self.ShapeAfterConv(conv1_shape, 0, 4, self.stride)
        conv3_shape = self.ShapeAfterConv(conv2_shape, 0, 4, self.stride)
        conv4_shape = self.ShapeAfterConv(conv3_shape, 0, 4, self.stride)
        embed_size = 8 * self.depth * np.prod(conv4_shape).item()
        return embed_size


class ObservationDecoder(nn.Module):

    def __init__(self, stochastic_size, deterministic_size, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.shape =(3,1000,3750)
        self.FullyConnected = nn.Linear(stochastic_size + deterministic_size, embedding_size)
        self.Conv1 = nn.ConvTranspose2d(256, 128, 4, stride=2)
        self.Conv2 = nn.ConvTranspose2d(128, 64, 4, stride=2)
        self.Conv3 = nn.ConvTranspose2d(64, 32, 4, stride=2)
        self.Conv4 = nn.ConvTranspose2d(32, 3, 4, stride=2)

    def forward(self, x):
        batch_shape = x.shape[:-1]
        embed_size = x.shape[-1]
        squeezed_size = np.prod(batch_shape).item()
        x = x.reshape(squeezed_size, embed_size)
        AfterLinear = F.elu(self.FullyConnected(x))

        AfterLinear = torch.reshape(AfterLinear,(-1, 256, 39, 152))
        AfterConv1 = F.elu(self.Conv1(AfterLinear))
        AfterConv2 = F.elu(self.Conv2(AfterConv1))
        AfterConv3 = F.elu(self.Conv3(AfterConv2))

        observation =self.Conv4(AfterConv3)
        trans = transforms.Compose([transforms.Resize(size=(666, 2475))])

        observation = trans(observation)

        obs_dist = dist.Independent(dist.Normal(observation, 1), len(self.shape))
        return obs_dist

    @staticmethod
    def ShapeAfterConv(h_in, padding, kernel_size, stride):
        ShapeAfterCNN = list()
        for x in h_in:
            ShapeAfterCNN.append(int((x + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.))
        return tuple(ShapeAfterCNN)
