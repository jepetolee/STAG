import snntorch as snn
import torch.nn as nn
import torch.nn.functional as F
import snntorch.surrogate as surrogate
import PIL
import torch
from torchvision import transforms
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, input_channel, output_channel, base_width=64,
                 downsample=None, groups=1, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        width = output_channel * (base_width // 64) * groups
        self.conv1 = nn.Conv2d(input_channel, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.Lif1 = snn.Leaky(beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25))

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False,
                               dilation=dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.Lif2 = snn.Leaky(beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25))

        self.conv3 = nn.Conv2d(width, output_channel * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channel * self.expansion)
        self.Lif3 = snn.Leaky(beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25))

        self.Lif4 = snn.Leaky(beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, inputs):
        identity = inputs
        membrane1 = self.Lif1.init_leaky()
        membrane2 = self.Lif2.init_leaky()
        membrane3 = self.Lif3.init_leaky()

        x = self.conv1(inputs)
        x = self.bn1(x)
        x, membrane1 = self.Lif1(x, membrane1)

        x = self.conv2(x)
        x = self.bn2(x)
        x, membrane2 = self.Lif2(x, membrane2)

        x = self.conv3(x)
        x = self.bn3(x)
        x, membrane3 = self.Lif3(x, membrane3)

        if self.downsample is not None:
            identity = self.downsample(inputs)
        x += identity
        out = self.Lif4(x)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers,zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=3, padding=3,
                               bias=False)
        self.bn1 =  nn.BatchNorm2d(self.inplanes)
        self.LIF = snn.Leaky(beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 6, layers[0])
        self.layer2 = self._make_layer(block, 8, layers[1], stride=3,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 10, layers[2], stride=3,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 12, layers[3], stride=3,
                                       dilate=replace_stride_with_dilation[2])
        self.layer5 = self._make_layer(block, 14, layers[4], stride=3,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):

        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, stride=stride, kernel_size=1,bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride,downsample= downsample, groups= self.groups,
                            base_width=self.base_width, dilation=previous_dilation))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.LIF(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
url = 'G:/ImgDataStorage/BTCUSDT/COMBINED/1.jpg'
crypto_chart = PIL.Image.open(url)
crypto_chart = trans(crypto_chart).float().cuda()
model = ResNet(Bottleneck, [3, 4, 6,8, 3]).cuda()
print(model(crypto_chart.reshape(-1,3,3000, 2400)).shape)