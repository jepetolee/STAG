import torch.nn as nn
from .BasicModel.RESNET import ResNet, Bottleneck

class TradingA2C(nn.Module):
    def __init__(self):
        super(TradingA2C, self).__init__()
        self.feature_extract = ResNet(Bottleneck, [3, 4, 6, 8])
        self.PiNet = VideoClassifier(output=3)
        self.DecidingNet = VideoClassifier(output=2)
        self.ValueNet = nn.Sequential(nn.GELU(),
                                      nn.Linear(3,1))
        self.softmax = nn.Softmax(dim=1)

    def Pi(self, input):
        x = self.feature_extract(input)
        x = x.view(-1, 3840)
        x = self.PiNet(x)
        return self.softmax(x)
    def Deciding(self, input):
        x = self.feature_extract(input)
        x = x.view(-1, 3840)
        x = self.DecidingNet(x)
        return self.softmax(x)

    def Value(self, input):
        x = self.Pi(input)
        return self.ValueNet(x)


class VideoClassifier(nn.Module):
    def __init__(self,output):
        super(VideoClassifier, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(3840,320),
                                        nn.GELU(),
                                        nn.Linear(320, 32),
                                        nn.GELU(),
                                        nn.Linear(32,output))
    def forward(self, input):
        return  self.classifier(input)
