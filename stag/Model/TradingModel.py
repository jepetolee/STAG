import torch.nn as nn
from .BasicModel.RESNET import ResNet, Bottleneck
import torch


class TradingA2C(nn.Module):
    def __init__(self):
        super(TradingA2C, self).__init__()
        self.feature_extract = ResNet(Bottleneck, [3, 4, 6, 8,10])
        self.PiNet = VideoClassifier(output=3)
        self.DecidingNet = nn.Sequential(nn.Linear(1600, 400),
                                        nn.GELU(),
                                         nn.Linear(400, 100),
                                         nn.GELU(),
                                        nn.Linear(100, 29))
        self.DecidingClassifier = nn.Sequential(nn.GELU(),
                                        nn.Linear(32, 2))
        self.ValueNet = nn.Sequential(nn.Linear(1600, 400),
                                        nn.GELU(),
                                        nn.Linear(400, 100),
                                        nn.GELU(),
                                        nn.Linear(100, 29))
        self.ValueClassifier = nn.Sequential(
            nn.GELU(),
            nn.Linear(32, 1))
        self.Trade_tensor = torch.zeros(1,800)
        self.softmax = nn.Softmax(dim=1)

    def Pi(self, input):
        x = self.feature_extract(input)
        x = x.view(-1, 800)
        self.Trade_tensor = x.detach()
        x = self.PiNet(x)
        return self.softmax(x)

    def Value(self, input,action,):
        x = self.feature_extract(input)
        x = x.view(-1, 800)
        x = torch.cat([x, self.Trade_tensor], dim=1)
        x = self.ValueNet(x)
        x = torch.cat([x,action],dim=1)
        return self.ValueClassifier(x)


    def Deciding(self, input,action):
        x = self.feature_extract(input)
        x = x.view(-1, 800)
        x = torch.cat([x, self.Trade_tensor], dim=1)
        x = self.DecidingNet(x)
        x = torch.cat([x, action], dim=1)
        x = self.DecidingClassifier(x)
        return self.softmax(x)


class VideoClassifier(nn.Module):
    def __init__(self, input =800, output=3):
        super(VideoClassifier, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(input, 200),
                                        nn.GELU(),
                                        nn.Linear(200, 32),
                                        nn.GELU(),
                                        nn.Linear(32, output))

    def forward(self, input):
        return self.classifier(input)
