import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, input_channel, output_channel, base_width=64,
                 downsample=None, groups=1, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        width = output_channel * (base_width // 64) * groups
        self.conv1 = nn.Conv2d(input_channel, width, kernel_size=1, stride=1, bias=False)
        self.gelu = nn.GELU()

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False,
                               dilation=dilation)

        self.conv3 = nn.Conv2d(width, output_channel * self.expansion, kernel_size=1, stride=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs):
        identity = inputs
        x = self.conv1(inputs)
        x = self.gelu(x)

        x = self.conv2(x)
        x = self.gelu(x)

        x = self.conv3(x)
        x = self.gelu(x)

        if self.downsample is not None:
            identity = self.downsample(inputs)
        x += identity
        out = self.gelu(x)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
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
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=3, padding=3,
                               bias=False)
        self.conv2 = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=5, stride=3, padding=3,
                               bias=False)
        self.gelu = nn.GELU()

        self.layer1 = self._make_layer(block, 2, layers[0])
        self.layer2 = self._make_layer(block, 3, layers[1], stride=3,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 4, layers[2], stride=3,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 5, layers[3], stride=3,
                                       dilate=replace_stride_with_dilation[1])
        self.layer5 = self._make_layer(block, 6, layers[4], stride=3,
                                       dilate=replace_stride_with_dilation[1])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

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
                nn.Conv2d(self.inplanes, planes * block.expansion, stride=stride, kernel_size=1, bias=False),
            )

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample, groups=self.groups,
                        base_width=self.base_width, dilation=previous_dilation)]

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
