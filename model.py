import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, last=False):
        super(ResidualBlock, self).__init__()
        out_channels = out_channels or in_channels
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.activate1 = nn.LeakyReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.residual_block = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activate2 = nn.Tanh() if last else nn.LeakyReLU(inplace=True)

    def forward(self, x):
        residual = self.residual_block(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activate1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.activate2(out)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        if in_channels!=out_channels:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                                      stride=2,padding=1),
                                            nn.ReLU(inplace=True))
        else:
            self.downsample = None
        self.block = ResidualBlock(out_channels, out_channels)

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels,combine='concat'):
        super(UpBlock, self).__init__()
        if in_channels!=out_channels:
            self.up = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels,
                                                       kernel_size=2, stride=2),
                                    nn.ReLU(inplace=True))
        else:
            self.up = None

        if combine=='concat':
            self.block = ResidualBlock(out_channels * 2, out_channels)
        elif combine=='add':
            self.block = ResidualBlock(out_channels, out_channels)
        self.combine = combine 

    def forward(self, x, x_skip):
        if self.up is not None:
            x = self.up(x)
        if self.combine=='concat':
            return self.block(torch.cat([x, x_skip], dim=1))
        elif self.combine=='add':
            return self.block(x + x_skip)
        


class UNet(nn.Module):
    def __init__(self, n_channels=[3, 64, 128, 256, 512, 1024]):
        super(UNet, self).__init__()
        self.inc = ResidualBlock(n_channels[0], n_channels[1])

        self.down_blocks = nn.ModuleList([
            DownBlock(n_channels[i], n_channels[i + 1])
            for i in range(1,len(n_channels) - 1)
        ])

        self.middle = ResidualBlock(n_channels[-1], n_channels[-1])

        self.up_blocks = nn.ModuleList([
            UpBlock(n_channels[i], n_channels[i - 1])
            for i in range(len(n_channels) - 1, 1, -1)
        ])

        self.outc = ResidualBlock(n_channels[1], n_channels[0], last=True)

    def forward(self, x):
        x = self.inc(x)
        x_skips = []
        for down_block in self.down_blocks:
            x_skips.append(x)
            x = down_block(x)
        x = self.middle(x)
        for up_block, x_skip in zip(self.up_blocks, x_skips[::-1]):
            x = up_block(x, x_skip)
        x = self.outc(x)
        return x

if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    model = UNet()
    x = torch.randn(1, 3, 1024, 1024)
    # print(model(x).shape)
    writer.add_graph(model, x)
    writer.close()
