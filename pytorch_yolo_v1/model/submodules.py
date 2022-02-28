import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d_bn_relu(in_dim, out_dim, kernel, stride=1, pad=0, dilate=1, group=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel, stride, pad, dilate, group),
        nn.BatchNorm2d(out_dim),
        nn.ReLU()
    )

def up_conv2d(in_dim, out_dim, kernel=3, pad=1, up_scale=2):
    return nn.Sequential(
        nn.Upsample(scale_factor=up_scale, mode='nearest'),
        nn.Conv2d(in_dim, out_dim, kernel, padding=pad)
    )

class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, act=True):
        super(Conv, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()
        )

    def forward(self, x):
        return self.convs(x)

class SPP(nn.Module):
    """
        Spatial Pyramid Pooling
    """
    def __init__(self):
        super(SPP, self).__init__()

    def forward(self, x):
        x_1 = torch.nn.functional.max_pool2d(x, 5, stride=1, padding=2)
        x_2 = torch.nn.functional.max_pool2d(x, 9, stride=1, padding=4)
        x_3 = torch.nn.functional.max_pool2d(x, 13, stride=1, padding=6)
        x = torch.cat([x, x_1, x_2, x_3], dim=1)

        return x


