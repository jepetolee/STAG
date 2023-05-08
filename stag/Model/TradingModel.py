import torch.nn as nn
from .BasicModel.RESNET import ResNet, Bottleneck
import torch


class TradingA2C(nn.Module):
    def __init__(self):
        super(TradingA2C, self).__init__()
        self.feature_extract = ResNet(Bottleneck, [3, 4, 6, 8,10])
        self.PiNet = VideoClassifier(480,3)
        self.DecidingNet = VideoClassifier(480,2)
        self.ValueNet = VideoClassifier(480,1)
        self.ValuePNet = VideoClassifier(480, 1)
        self.softmax = nn.Softmax(dim=1)

    def Pi(self, input):
        x = self.feature_extract(input)
        x = x.view(-1, 480)
        x = self.PiNet(x)
        return self.softmax(x)

    def Value(self, input):
        x = self.feature_extract(input)
        x = x.view(-1, 480)
        return self.ValueNet(x)

    def ValueP(self, input):
        x = self.feature_extract(input)
        x = x.view(-1, 480)
        return self.ValuePNet(x)

    def Deciding(self, input):
        x = self.feature_extract(input)
        x = x.view(-1, 480)
        x = self.DecidingNet(x)
        return self.softmax(x)


class VideoClassifier(nn.Module):
    def __init__(self, input, output):
        super(VideoClassifier, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(input, 160),
                                        nn.GELU(),
                                        nn.Linear(160, 32),
                                        nn.GELU(),
                                        nn.Linear(32, output))

    def forward(self, input):
        return self.classifier(input)
