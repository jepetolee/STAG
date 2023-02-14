import torch
from torch import nn, jit


class BasicEncoderBlock(nn.Module):
    def __init__(self, inputs, depth):
        super().__init__()
        self.depth = depth
        self.residual1 = nn.Sequential(nn.Conv2d(inputs, depth, 1, stride=2, bias=False), nn.BatchNorm2d(3), nn.ELU())
        self.conv1 = nn.Sequential(nn.BatchNorm2d(inputs), nn.ELU(), nn.Conv2d(inputs, depth, 3, stride=2))
        self.conv2 = nn.Sequential(nn.BatchNorm2d(inputs), nn.ELU(), nn.Conv2d(depth, depth, 3, stride=2))

    def forward(self, image):
        skip = image
        if image.shape[-1] != self.depth:
            skip = self.residual1(image)
        x = self.conv1(image)
        x = self.conv2(x)
        return skip + 0.1 * x


class ObservationResnetEncoder(nn.Module):
    def __init__(self, stride=2, shape=(3, 1000, 3750)):
        super().__init__()
        self.ConvBlock1 = BasicEncoderBlock(3, 32)
        self.AvgPool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.ConvBlock2 = BasicEncoderBlock(32, 64)
        self.AvgPool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.ConvBlock3 = BasicEncoderBlock(64, 128)
        self.AvgPool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.ConvBlock4 = BasicEncoderBlock(128, 256)
        self.AvgPool4 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.shape = shape
        self.depth = 32
        self.stride = stride
        self.FCLayer = nn.Identity()

    def forward(self, observation):
        conv1 = self.ConvBlock1(observation)
        conv1 = self.AvgPool1(conv1)
        conv2 = self.ConvBlock2(conv1)
        conv2 = self.AvgPool2(conv2)
        conv3 = self.ConvBlock3(conv2)
        conv3 = self.AvgPool1(conv3)
        conv4 = self.ConvBlock4(conv3)
        conv4 = self.AvgPool1(conv4)
        Output = self.FCLayer(conv4)
        Output = torch.reshape(Output, (Output.shape[0], -1))
        return Output



