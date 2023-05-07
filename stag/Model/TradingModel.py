import torch.nn as nn
from .BasicModel.RESNET import ResNet, Bottleneck
import torch


class TradingA2C(nn.Module):
    def __init__(self):
        super(TradingA2C, self).__init__()
        self.feature_extract = ResNet(Bottleneck, [3, 4, 6, 8])
        self.PiNet = VideoClassifier(output=3)
        self.DecidingNet = VideoClassifier(output=2)
        self.ValueNet =VideoClassifier(output=1)
        self.softmax = nn.Softmax(dim=1)

    def Pi(self, input):
        x = self.feature_extract(input)
        x = x.view(-1, 1680)
        x = self.PiNet(x)
        return self.softmax(x)

    def Value(self, input):
        x = self.feature_extract(input)
        x = x.view(-1, 1680)
        return self.ValueNet(x)


    def Deciding(self, input):
        x = self.feature_extract(input)
        x = x.view(-1, 1680)
        x = self.DecidingNet(x)
        return self.softmax(x)


class VideoClassifier(nn.Module):
    def __init__(self, input=1680, output=3):
        super(VideoClassifier, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(input, 280),
                                        nn.GELU(),
                                        nn.Linear(280, 32),
                                        nn.GELU(),
                                        nn.Linear(32, output))

    def forward(self, input):
        return self.classifier(input)
