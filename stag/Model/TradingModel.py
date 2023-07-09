import torch.nn as nn
from .BasicModel.Moe import MixedActor
from .BasicModel.DCGN import *
import torch




class TradingA2C(nn.Module):
    def __init__(self,intersize = 8800):
        super(TradingA2C, self).__init__()
        self.intersize = intersize
        self.feature_extract = nn.Sequential(nn.Flatten(),
                                             nn.Linear(8800,2200),
                                             nn.GELU(),
                                             nn.Linear(2200,550),
                                             nn.GELU())

        self.PiNet = VideoClassifier(intersize,3)
        self.DecidingNet = VideoClassifier(intersize,2)
        self.ValueNet = VideoClassifier(intersize,1)
        self.ValuePNet = VideoClassifier(intersize, 1)
        self.softmax = nn.Softmax(dim=1)

        self.in_size = 550
        self.h1 = 110
        self.c_t1_position = None
        self.h_t1_position = None
        self.c_t1_decide = None
        self.h_t1_decide = None
        self.lstmPi = nn.LSTMCell(input_size=self.in_size, hidden_size=self.h1)
        self.lstmPiValue = nn.LSTMCell(input_size=self.in_size, hidden_size=self.h1)
        self.lstmDecide = nn.LSTMCell(input_size=self.in_size, hidden_size=self.h1)
        self.lstmDecideValue = nn.LSTMCell(input_size=self.in_size, hidden_size=self.h1)

    def reset_lstm(self, buf_size=80):
        with torch.no_grad():
            self.h_t1_decide =self.c_t1_decide =self.h_t1_position = self.c_t1_position = torch.zeros(buf_size, self.h1, device=self.lstm.weight_ih.device)
            self.h_t1_decideV = self.c_t1_decideV = self.h_t1_positionV = self.c_t1_positionV = torch.zeros(buf_size,self.h1,device=self.lstm.weight_ih.device)

    def Pi(self, input):
        x = self.feature_extract(input)
        h_t1_position, c_t1_position = self.lstmPi(x, (self.h_t1_position, self.c_t1_position))
        self.h_t1_position, self.c_t1_position= h_t1_position, c_t1_position
        x = h_t1_position.view(-1, self.intersize)
        x = self.PiNet(x)
        return self.softmax(x)

    def ValueP(self, input):
        x = self.feature_extract(input)
        h_t1_position, c_t1_position = self.lstmPiValue(x, (self.h_t1_position, self.c_t1_position))
        self.h_t1_positionV, self.c_t1_positionV =h_t1_position, c_t1_position
        x = h_t1_position.view(-1, self.intersize)
        return self.ValuePNet(x)

    def Deciding(self, input):
        x = self.feature_extract(input)
        h_t1_decide, c_t1_decide  =  self.lstmDecide(x, (self.h_t1_decide, self.c_t1_decide))
        self.h_t1_decide, self.c_t1_decide =  h_t1_decide, c_t1_decide
        x = h_t1_decide.view(-1, self.intersize)
        x = self.DecidingNet(x)
        return self.softmax(x)
    def Value(self, input):
        x = self.feature_extract(input)
        h_t1_decideV,c_t1_decideV  = self.lstmDecideValue(x, (self.h_t1_decide, self.c_t1_decide))
        self.h_t1_decideV, self.c_t1_decideV= h_t1_decideV,c_t1_decideV
        x =  h_t1_decideV.view(-1, self.intersize)
        return self.ValueNet(x)


class VideoClassifier(nn.Module):
    def __init__(self, input, output):
        super(VideoClassifier, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(input,2200),
                                        nn.GELU(),
                                        nn.Linear(2200, 440),
                                        nn.GELU(),
                                        nn.Linear(440, 32),
                                        nn.GELU(),
                                        nn.Linear(32, output))

    def forward(self, input):
         return self.classifier(input)


class TradingLSTMA2C(nn.Module):
    def __init__(self,intersize = 8800):
        super(TradingLSTMA2C, self).__init__()
        self.feature_extract = nn.LSTMCell(8800,1100)
        self.linear =  nn.Sequential(nn.Flatten(),
                                     nn.Linear(1100,110))
        self.intersize = intersize
        self.PiNet = MixedActor(3)
        self.ValuePNet = MixedActor(1)

        self.DecidingNet = MixedActor(2)
        self.ValueNet = MixedActor(1)
        self.softmax = nn.Softmax(dim=1)

    def reset_lstm(self):
        with torch.no_grad():
            self.h_t1_decide = self.c_t1_decide = self.h_t1_position = self.c_t1_position \
                = self.h_t1_positionV = self.c_t1_positionV = torch.zeros(80, 1100, device='cuda')
    def del_lstm(self):
        del self.c_t1_position, self.c_t1_decide
        del self.h_t1_position, self.h_t1_decide
    def Pi(self, input):
        input = input.reshape(-1,8800)
        h_t1_position, c_t1_position = self.feature_extract(input,(self.h_t1_position, self.c_t1_position))
        self.h_t1_position, self.c_t1_position= h_t1_position.clone().detach() , c_t1_position.clone().detach()
        x= self.linear(h_t1_position)
        x = x.reshape(-1, self.intersize)
        x = self.PiNet(x)
        return self.softmax(x)

    def ValueP(self, input):
        input = input.reshape(-1, 8800)
        h_t1_positionV, c_t1_positionV = self.feature_extract(input,(self.h_t1_positionV, self.c_t1_positionV))
        self.h_t1_positionV, self.c_t1_positionV = h_t1_positionV.clone().detach(), c_t1_positionV.clone().detach()
        x = self.linear(h_t1_positionV)
        x = x.reshape(-1, self.intersize)
        return self.ValuePNet(x)

    def Deciding(self, input):
        input = input.reshape(-1, 8800)
        h_t1_decide, c_t1_decide = self.feature_extract(input,(self.h_t1_decide, self.c_t1_decide))
        x = self.linear(h_t1_decide)
        self.h_t1_decide, self.c_t1_decide = h_t1_decide.clone().detach(), c_t1_decide.clone().detach()
        x = x.reshape(-1, self.intersize)
        x = self.DecidingNet(x)
        return self.softmax(x)
    def Value(self, input):
        input = input.reshape(-1, 8800)
        h_t1_decideV, c_t1_decideV= self.feature_extract(input,(self.h_t1_position, self.c_t1_position))
        x= self.linear(h_t1_decideV)
        x =  x.reshape(-1, self.intersize)
        return self.ValueNet(x)


class TradingConvLSTMA2C(nn.Module):
    def __init__(self,intersize = 880):
        super(TradingConvLSTMA2C, self).__init__()
        self.conv = nn.Sequential(nn.Conv1d(84,35,3,stride=1),
                                  nn.BatchNorm1d(35),
                                  nn.Conv1d(35, 35, 3, stride=2),
                                  nn.LeakyReLU(),
                                  nn.Conv1d(35,11,3,stride=1),
                                  nn.BatchNorm1d(11),
                                  nn.Conv1d(11, 11, 3, stride=2),
                                  nn.LeakyReLU(),
                                  nn.Conv1d(11, 5, 3, stride=1),
                                  nn.BatchNorm1d(5),
                                  nn.Conv1d(5, 5, 3, stride=2),
                                  nn.LeakyReLU())
        self.feature_extract = nn.LSTMCell(110,11)
        self.intersize = intersize
        self.PiNet = MixedActor(intersize,8,3)
        self.ValuePNet = MixedActor(intersize*3,24,1)

        self.DecidingNet = MixedActor(intersize,8,2)
        self.ValueNet = MixedActor(intersize*3,24,1)
        self.softmax = nn.Softmax(dim=1)

    def reset_lstm(self):
        with torch.no_grad():
            self.h_t1_decide = self.c_t1_decide = self.h_t1_position = self.c_t1_position \
                = self.h_t1_positionV = self.c_t1_positionV =self.h_t1_decideV=self.c_t1_decideV=\
                torch.zeros(80, 11, device='cuda')
    def del_lstm(self):
        del self.c_t1_position, self.c_t1_decide
        del self.h_t1_position, self.h_t1_decide
    def Pi(self, input):
        x = self.conv(input.permute(0,2,1))
        x = x.reshape(-1,110)
        h_t1_position, c_t1_position = self.feature_extract(x,(self.h_t1_position, self.c_t1_position))
        self.h_t1_position, self.c_t1_position= h_t1_position.clone().detach() , c_t1_position.clone().detach()
        x = h_t1_position.reshape(-1, self.intersize)
        x = self.PiNet(x)
        return self.softmax(x)

    def ValueP(self, input):
        x = self.conv(input.permute(0, 2, 1))
        x = x.reshape(-1, 110)
        h_t1_positionV, c_t1_positionV = self.feature_extract(x,(self.h_t1_positionV, self.c_t1_positionV))
        self.h_t1_positionV, self.c_t1_positionV = h_t1_positionV.clone().detach(), c_t1_positionV.clone().detach()
        h_t1_position =self.h_t1_position
        h_t1_decide =self.h_t1_decide
        x = torch.stack([h_t1_positionV,h_t1_position,h_t1_decide])
        x = x.reshape(-1, self.intersize*3)
        return self.ValuePNet(x)

    def Deciding(self, input):
        x = self.conv(input.permute(0, 2, 1))
        x = x.reshape(-1, 110)
        h_t1_decide, c_t1_decide = self.feature_extract(x,(self.h_t1_decide, self.c_t1_decide))
        self.h_t1_decide, self.c_t1_decide = h_t1_decide.clone().detach(), c_t1_decide.clone().detach()
        x = h_t1_decide.reshape(-1, self.intersize)
        x = self.DecidingNet(x)
        return self.softmax(x)
    def Value(self, input):
        x = self.conv(input.permute(0, 2, 1))
        x = x.reshape(-1, 110)
        h_t1_decideV, c_t1_decideV= self.feature_extract(x,(self.h_t1_decideV, self.c_t1_decideV))
        self.h_t1_decideV, self.c_t1_decideV = h_t1_decideV.clone().detach(), c_t1_decideV.clone().detach()
        h_t1_position =self.h_t1_position
        h_t1_decide =self.h_t1_decide
        x = torch.stack([h_t1_decideV,h_t1_position,h_t1_decide])
        x =  x.reshape(-1, self.intersize*3)
        return self.ValueNet(x)

