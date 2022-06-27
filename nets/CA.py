import torch
from torch import nn


class CAtt(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CAtt, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))  # 输出size（h, 1）
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))  # 输出size(1, w)

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=(1, 1))
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)

        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=(1, 1), stride=(1, 1))
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=(1, 1), stride=(1, 1))

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x0):

        x_h = self.avg_pool_x(x0).permute(0, 1, 3, 2)  # b, c, w(=1), h
        x_w = self.avg_pool_y(x0)  # b, c, h(=1), w

        y0 = torch.cat([x_h, x_w], 3)
        y0 = self.conv_1x1(y0)
        y0 = self.bn(y0)
        y0 = self.relu(y0)

        y_h, y_w = y0.split([self.h, self.w], 3)
        y_h = y_h.permute(0, 1, 3, 2)
        y_h = self.F_h(y_h)
        y_w = self.F_w(y_w)

        s_h = self.sigmoid_h(y_h)
        s_w = self.sigmoid_w(y_w)

        out = x0 * s_h.expand_as(x0) * s_w.expand_as(x0)

        return out
