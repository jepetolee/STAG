import math, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F


def get_dwconv(dim, kernel, bias):
    return nn.Conv1d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)


class RecConv(nn.Module):
    def __init__(self, dim, order=3):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv1d(dim, 2 * dim, 1)

        self.dwconv = get_dwconv(sum(self.dims), 3, True)

        self.proj_out = nn.Conv1d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv1d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)])

    def forward(self, x):
        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc)

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa + dw_list[0]

        for i in range(self.order - 1):
            x = self.pws[i](x) + dw_list[i + 1]

        x = self.proj_out(x)

        return x


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class SERecBlock(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(SERecBlock, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = RecConv(width)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.convs(out)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out


class SERecLayer(nn.Module):

    def __init__(self, num_blocks, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(SERecLayer, self).__init__()
        self.SERecBlocks = [SERecBlock(inplanes, planes, kernel_size=kernel_size, dilation=dilation, scale=scale) for _
                            in range(num_blocks)]

    def forward(self, x):
        for block in self.SERecBlocks:
            x = block(x)
        return x


class ECAPA_PP(nn.Module):

    def __init__(self, Channels=[512, 256, 128, 64], Block=[8, 24, 8], device='cuda'):
        super(ECAPA_PP, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80),
        )

        self.conv1 = nn.Conv1d(80, Channels[0], kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(Channels[0])
        self.layer1 = SERecLayer(Block[0], Channels[0], Channels[1], kernel_size=3, dilation=2, scale=8)
        self.layer2 = SERecLayer(Block[1], Channels[1], Channels[2], kernel_size=3, dilation=3, scale=8)
        self.layer3 = SERecLayer(Block[2], Channels[2], Channels[3], kernel_size=3, dilation=4, scale=8)

        self.DWCONV_L = [get_dwconv(Channels[i], 3, True) for i in range(4)]
        self.DWCONV_P = [get_dwconv(Channels[i + 2], 3, True) for i in range(2)]
        self.WeightP = [nn.Parameter(torch.zeros(1, 2)).to(device) for _ in range(2)]
        self.DWCONV_T = [get_dwconv(Channels[i], 3, True) for i in range(4)]
        self.WeightT = [nn.Parameter(torch.zeros(1, 3)).to(device) for _ in range(4)]
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(Channels[3], 64, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Tanh(),  # I add this layer
            nn.Conv1d(64, 128, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(256)
        self.fc6 = nn.Linear(256, 192)
        self.bn6 = nn.BatchNorm1d(192)

    def forward(self, x):
        with torch.no_grad():
            x = self.torchfbank(x) + 1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)

        x = self.conv1(x)
        x = self.relu(x)
        f1 = self.bn1(x)

        # Top Down
        f2 = self.layer1(f1)
        f3 = self.layer2(f2)
        f4 = self.layer3(f3)
        # Lateral
        L1 = self.DWCONV_L[0](f1)
        L2 = self.DWCONV_L[1](f2)
        L3 = self.DWCONV_L[2](f3)
        L4 = self.DWCONV_L[3](f4)
        # Top Down
        IP3 = F.softmax(self.WeightP[0], dim=1)
        P3 = self.DWCONV_P[0](torch.sum(torch.dot(IP3, torch.stack([L4, L3], dim=0)), dim=0))  # W*L4 + W*L3
        IP2 = F.softmax(self.WeightP[1], dim=1)
        P2 = self.DWCONV_P[1](torch.sum(torch.dot(IP2, torch.stack([P3, L2], dim=0)), dim=0))  # W*P3 + W*L2
        # BottomUp

        IT1 = F.softmax(self.WeightT[0], dim=1)
        T1 = self.DWCONV_T[0](
            torch.sum(torch.dot(IT1, torch.stack([L1, P2, torch.zeros_like(P2).cuda()], dim=0)), dim=0))
        IT2 = F.softmax(self.WeightT[1], dim=1)
        T2 = self.DWCONV_T[1](torch.sum(torch.dot(IT2, torch.stack([L2, P2, T1], dim=0)), dim=0))
        IT3 = F.softmax(self.WeightT[2], dim=1)
        T3 = self.DWCONV_T[2](torch.sum(torch.dot(IT3, torch.stack([L3, P3, T2], dim=0)), dim=0))
        IT4 = F.softmax(self.WeightT[3], dim=1)
        T4 = self.DWCONV_T[3](
            torch.sum(torch.dot(IT4, torch.stack([L4, torch.zeros_like(L4).cuda(), T3], dim=0)), dim=0))

        x = self.layer4(T4)
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                              torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)), dim=1)

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))

        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x
