import torch
from torch import nn, jit
from torch.nn import functional as F
import numpy as np


class ObservationEncoder(jit.ScriptModule):
    def __init__(self):
        super().__init__()
        self.Conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.Conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.Conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.Conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.FCLayer = nn.Identity()

    @jit.script_method
    def forward(self, observation):
        Hidden1 = F.elu(self.Conv1(observation))
        Hidden2 = F.elu(self.Conv2(Hidden1))
        Hidden3 = F.elu(self.Conv2(Hidden2))
        Hidden4 = F.elu(self.Conv4(Hidden3))
        Output = self.FCLayer(Hidden4)
        return Output

    @staticmethod
    def ShapeAfterConv(h_in, padding, kernel_size, stride):
        ShapeAfterCNN = list()
        for x in h_in:
            ShapeAfterCNN.append(int((x + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.))
        return tuple(ShapeAfterCNN)

    @property
    def embed_size(self):
        conv1_shape = self.ShapeAfterConv(self.shape[1:], 0, 4, self.stride)
        conv2_shape = self.ShapeAfterConv(conv1_shape, 0, 4, self.stride)
        conv3_shape = self.ShapeAfterConv(conv2_shape, 0, 4, self.stride)
        conv4_shape = self.ShapeAfterConv(conv3_shape, 0, 4, self.stride)
        embed_size = 8 * self.depth * np.prod(conv4_shape).item()
        return embed_size


class ObservationDecoder(jit.ScriptModule):
    __constants__ = ['embedding_size']

    def __init__(self, belief_size, state_size, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.FullyConnected = nn.Linear(belief_size + state_size, embedding_size)
        self.Conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
        self.Conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.Conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.Conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)
        self.modules = [self.FullyConnected, self.conv1, self.conv2, self.conv3, self.conv4]

    @jit.script_method
    def forward(self, belief, state):
        AfterFullyConnected = self.FullyConnected(torch.cat([belief, state], dim=1))
        Flatten = AfterFullyConnected.view(-1, self.embedding_size, 1, 1)
        AfterConv1 = F.elu(self.Conv1(Flatten))
        AfterConv2 = F.elu(self.Conv2(AfterConv1))
        AfterConv3 = F.elu(self.Conv3(AfterConv2))
        observation = self.Conv4(AfterConv3)
        return observation
