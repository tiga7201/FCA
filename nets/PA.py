from torch import nn


class PAtt(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(PAtt, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_h = nn.AdaptiveAvgPool2d((h, 1))  # output size = (h, 1）
        self.max_pool_h = nn.AdaptiveMaxPool2d((h, 1))
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, w))  # output size = (1, w)
        self.max_pool_w = nn.AdaptiveMaxPool2d((1, w))

        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=(1, 1)),
            nn.BatchNorm2d(channel//reduction),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=(1, 1))
        )

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        x_h_avg = self.avg_pool_h(x)
        x_h_max = self.max_pool_h(x)
        x_w_avg = self.avg_pool_w(x)
        x_w_max = self.max_pool_w(x)

        y_h_avg = self.shared_conv(x_h_avg)
        y_h_max = self.shared_conv(x_h_max)
        y_w_avg = self.shared_conv(x_w_avg)
        y_w_max = self.shared_conv(x_w_max)

        att_h = y_h_avg + y_h_max
        att_h = self.sigmoid_h(att_h)
        att_w = y_w_avg + y_w_max
        att_w = self.sigmoid_w(att_w)

        out = x * att_h.expand_as(x) * att_w.expand_as(x)

        return out
