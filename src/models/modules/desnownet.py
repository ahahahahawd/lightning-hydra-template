""" Reproduction of the DesnowNet model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Reduction_A(nn.Module):
    # 35 -> 17
    def __init__(self, in_channels, k, l, m, n):
        super(Reduction_A, self).__init__()
        self.branch_0 = Conv2d(in_channels, n, 3, stride=1, padding=1, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, k, 1, stride=1, padding=0, bias=False),
            Conv2d(k, l, 3, stride=1, padding=1, bias=False),
            Conv2d(l, m, 3, stride=1, padding=1, bias=False),
        )
        # kenel size and stride to 3 and 1 by DesnowNet
        self.branch_2 = nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        return torch.cat((x0, x1, x2), dim=1)  # 17 x 17 x 1024


class Stem(nn.Module):
    def __init__(self, in_channels):
        super(Stem, self).__init__()
        self.conv2d_1a_3x3 = Conv2d(in_channels, 16, 3, stride=1, padding=1, bias=False)

        self.conv2d_2a_3x3 = Conv2d(16, 16, 3, stride=1, padding=1, bias=False)
        self.conv2d_2b_3x3 = Conv2d(16, 32, 3, stride=1, padding=1, bias=False)

        # kenel size and stride to 3 and 1 by DesnowNet
        self.mixed_3a_branch_0 = nn.MaxPool2d(3, stride=1, padding=1)
        self.mixed_3a_branch_1 = Conv2d(32, 48, 3, stride=1, padding=1, bias=False)

        self.mixed_4a_branch_0 = nn.Sequential(
            Conv2d(80, 32, 1, stride=1, padding=0, bias=False),
            Conv2d(32, 48, 3, stride=1, padding=1, bias=False),
        )
        self.mixed_4a_branch_1 = nn.Sequential(
            Conv2d(80, 32, 1, stride=1, padding=0, bias=False),
            Conv2d(32, 32, (1, 7), stride=1, padding=(0, 3), bias=False),
            Conv2d(32, 32, (7, 1), stride=1, padding=(3, 0), bias=False),
            Conv2d(32, 48, 3, stride=1, padding=1, bias=False)
        )

        self.mixed_5a_branch_0 = Conv2d(96, 96, 3, stride=1, padding=1, bias=False)
        # kenel size and stride to 3 and 1 by DesnowNet
        self.mixed_5a_branch_1 = nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv2d_1a_3x3(x)  # 149 x 149 x 32
        x = self.conv2d_2a_3x3(x)  # 147 x 147 x 32
        x = self.conv2d_2b_3x3(x)  # 147 x 147 x 64
        x0 = self.mixed_3a_branch_0(x)
        x1 = self.mixed_3a_branch_1(x)
        x = torch.cat((x0, x1), dim=1)  # 73 x 73 x 160
        x0 = self.mixed_4a_branch_0(x)
        x1 = self.mixed_4a_branch_1(x)
        x = torch.cat((x0, x1), dim=1)  # 71 x 71 x 192
        x0 = self.mixed_5a_branch_0(x)
        x1 = self.mixed_5a_branch_1(x)
        x = torch.cat((x0, x1), dim=1)  # 35 x 35 x 384
        return x


class Inception_A(nn.Module):
    def __init__(self, in_channels):
        super(Inception_A, self).__init__()
        self.branch_0 = Conv2d(in_channels, 48, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False),
            Conv2d(32, 48, 3, stride=1, padding=1, bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False),
            Conv2d(32, 48, 3, stride=1, padding=1, bias=False),
            Conv2d(48, 48, 3, stride=1, padding=1, bias=False),
        )
        self.brance_3 = nn.Sequential(
            # remove by DesnowNet
            # nn.AvgPool2d(3, 1, padding=1, count_include_pad=False),
            Conv2d(192, 48, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.brance_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class Inception_B(nn.Module):
    def __init__(self, in_channels):
        super(Inception_B, self).__init__()
        self.branch_0 = Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 96, 1, stride=1, padding=0, bias=False),
            Conv2d(96, 112, (1, 7), stride=1, padding=(0, 3), bias=False),
            Conv2d(112, 128, (7, 1), stride=1, padding=(3, 0), bias=False),
        )
        self.branch_2 = nn.Sequential(
            Conv2d(in_channels, 96, 1, stride=1, padding=0, bias=False),
            Conv2d(96, 96, (7, 1), stride=1, padding=(3, 0), bias=False),
            Conv2d(96, 112, (1, 7), stride=1, padding=(0, 3), bias=False),
            Conv2d(112, 112, (7, 1), stride=1, padding=(3, 0), bias=False),
            Conv2d(112, 128, (1, 7), stride=1, padding=(0, 3), bias=False)
        )
        self.branch_3 = nn.Sequential(
            # remove by DesnowNet
            # nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            Conv2d(in_channels, 64, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class Reduction_B(nn.Module):
    # 17 -> 8
    def __init__(self, in_channels):
        super(Reduction_B, self).__init__()
        self.branch_0 = nn.Sequential(
            Conv2d(in_channels, 96, 1, stride=1, padding=0, bias=False),
            Conv2d(96, 96, 3, stride=1, padding=1, bias=False),
        )
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 128, 1, stride=1, padding=0, bias=False),
            Conv2d(128, 128, (1, 7), stride=1, padding=(0, 3), bias=False),
            Conv2d(128, 160, (7, 1), stride=1, padding=(3, 0), bias=False),
            Conv2d(160, 160, 3, stride=1, padding=1, bias=False)
        )
        # kenel size and stride to 3 and 1 by DesnowNet
        self.branch_2 = nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        return torch.cat((x0, x1, x2), dim=1)  # 8 x 8 x 1536


class Inception_C(nn.Module):
    def __init__(self, in_channels):
        super(Inception_C, self).__init__()
        self.branch_0 = Conv2d(in_channels, 128, 1, stride=1, padding=0, bias=False)

        self.branch_1 = Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False)
        self.branch_1_1 = Conv2d(192, 128, (1, 3), stride=1, padding=(0, 1), bias=False)
        self.branch_1_2 = Conv2d(192, 128, (3, 1), stride=1, padding=(1, 0), bias=False)

        self.branch_2 = nn.Sequential(
            Conv2d(in_channels, 182, 1, stride=1, padding=0, bias=False),
            Conv2d(182, 224, (3, 1), stride=1, padding=(1, 0), bias=False),
            Conv2d(224, 256, (1, 3), stride=1, padding=(0, 1), bias=False),
        )
        self.branch_2_1 = Conv2d(256, 128, (1, 3), stride=1, padding=(0, 1), bias=False)
        self.branch_2_2 = Conv2d(256, 128, (3, 1), stride=1, padding=(1, 0), bias=False)

        self.branch_3 = nn.Sequential(
            # remove by DesnowNet
            # nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            Conv2d(in_channels, 128, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x1_1 = self.branch_1_1(x1)
        x1_2 = self.branch_1_2(x1)
        x1 = torch.cat((x1_1, x1_2), 1)
        x2 = self.branch_2(x)
        x2_1 = self.branch_2_1(x2)
        x2_2 = self.branch_2_2(x2)
        x2 = torch.cat((x2_1, x2_2), dim=1)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)  # 8 x 8 x 1536


class Inceptionv4(nn.Module):
    def __init__(
        self,
        in_channels=3,
        # classes=1000,
        k=96, l=112, m=128, n=192
    ):
        super(Inceptionv4, self).__init__()
        blocks = []
        blocks.append(Stem(in_channels))
        for i in range(4):
            blocks.append(Inception_A(192))
        blocks.append(Reduction_A(192, k, l, m, n))
        for i in range(7):
            blocks.append(Inception_B(512))
        blocks.append(Reduction_B(512))
        for i in range(3):
            blocks.append(Inception_C(768))
        self.features = nn.Sequential(*blocks)
        # remove by DesnowNet
        # self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        # self.linear = nn.Linear(1536, classes)

    def forward(self, x):
        x = self.features(x)
        # remove by DesnowNet
        # x = self.global_average_pooling(x)
        # x = x.view(x.size(0), -1)
        # x = self.linear(x)
        return x


class Descriptor(nn.Module):
    """Descriptor"""

    def __init__(
        self,
        in_c,
        ):
        super().__init__()
        self.model = Inceptionv4(in_channels=in_c)

    def forward(self, x):
        pass
        return x


class RecoveryT(nn.Module):
    """Recovery Submodule"""

    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass
        return x


class RecoveryR(nn.Module):
    """Recovery Submodule"""

    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass
        return x


class DesnowNet(nn.Module):
    """DesnowNet"""

    def __init__(self):
        super().__init__()
        self.Dt = Descriptor(3)
        self.Dr = Descriptor()
        self.Rt = RecoveryT()
        self.Rr = RecoveryR()

    def forward(self, x):
        ft = self.Dt(x)
        fc, y = self.Rt(x, ft)
        fr = self.Dr(fc)
        r = self.Rr(fr)
        y_ = y + r
        return x


if __name__ == "__main__":
    x = torch.randn(1, 3, 64, 64)
    print(x.shape)
    model = Inceptionv4()
    # print(model)
    out = model(x)
    print(out.shape)
