import snntorch as snn
import torch.nn as nn
import torch.nn.functional as F
import snntorch.surrogate as surrogate


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

        self.Lif4 = snn.Leaky(beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25))
        self.downsample = downsample
        self.stride = stride
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.conv1.weight)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        torch.nn.init.xavier_uniform(self.conv3.weight)

    def call(self, inputs):
        identity = inputs
        membrane1 = self.Lif1.init_leaky()
        membrane2 = self.Lif2.init_leaky()
        membrane3 = self.Lif3.init_leaky()
        membrane4 = self.Lif4.init_leaky()

        x = self.conv1(inputs)
        x = self.bn1(x)
        x, membrane1 = self.Lif1(x, membrane1)

        x = self.conv2(x)
        x = self.bn2(x)
        x, membrane2 = self.Lif2(x, membrane2)

        x = self.conv2(x)
        x = self.bn2(x)
        x, membrane3 = self.Lif2(x, membrane3)

        if self.downsample is not None:
            identity = self.downsample(inputs)

        x += identity
        out, membrane4 = self.Lif4(x, membrane4)
        return out, membrane4

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
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
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x
