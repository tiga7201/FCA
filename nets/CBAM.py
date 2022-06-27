import torch
import torch.nn as nn


class ChA(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, (1, 1), bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, (1, 1), bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        ch_att = avg_out + max_out
        ch_att = self.sigmoid(ch_att)
        out1 = x * ch_att.expand_as(x)
        return out1


class SpA(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpA, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, (7, 7), padding=(kernel_size // 2, kernel_size // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sp_att = torch.cat([avg_out, max_out], dim=1)
        sp_att = self.conv1(sp_att)
        sp_att = self.sigmoid(sp_att)
        out2 = x * sp_att.expand_as(x)
        return out2
