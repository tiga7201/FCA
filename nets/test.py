from PA import PAtt
import torch

if __name__ == '__main__':
    x = torch.randn(1, 16, 128, 64)    # b, c, h, w
    ca_model = PAtt(channel=16, h=128, w=64)
    y = ca_model(x)
    print(y.shape)
