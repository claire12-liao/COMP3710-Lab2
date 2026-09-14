"""All assessed networks are defined here from layers, without pretrained models."""
import torch
from torch import nn
import torch.nn.functional as F


class FaceCNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        # DEMO P3-A: 用两个卷积层提取人脸的局部特征，每层有 32 个 3×3 滤波器。
        # 池化缩小特征图，最后由全连接层输出各个人物类别的分数。
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((6, 4)))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32 * 6 * 4, 128),
                                        nn.ReLU(), nn.Dropout(.4), nn.Linear(128, classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        # DEMO P3-B: 把输入的捷径分支与卷积处理后的结果相加，保留原有信息。
        # 这条较直接的路径有助于梯度向前面的层传播，使深层网络更容易训练。
        return F.relu(self.bn2(self.conv2(x)) + residual)


class ResNet18(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        # CIFAR images are 32x32: use a 3x3 stride-1 stem, without ImageNet's initial max pool.
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        layers = []
        incoming = 64
        for index, channels in enumerate([64, 128, 256, 512]):
            layers += [ResidualBlock(incoming, channels, 1 if index == 0 else 2),
                       ResidualBlock(channels, channels)]
            incoming = channels
        self.blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.fc(self.pool(self.blocks(self.stem(x))).flatten(1))


class VAE(nn.Module):
    def __init__(self, size=64, latent=16, base=32):
        super().__init__()
        if size % 16:
            raise ValueError("VAE size must be divisible by 16.")
        self.size, self.latent, self.base = size, latent, base
        channels = [1, base, base * 2, base * 4, base * 8]
        self.encoder = nn.Sequential(*[layer for a, b in zip(channels[:-1], channels[1:])
                                       for layer in (nn.Conv2d(a, b, 4, 2, 1), nn.ReLU())])
        self.flat_dim = base * 8 * (size // 16) ** 2
        self.mu = nn.Linear(self.flat_dim, latent)
        self.logvar = nn.Linear(self.flat_dim, latent)
        self.expand = nn.Linear(latent, self.flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(base, 1, 4, 2, 1), nn.Sigmoid())

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.mu(h), self.logvar(h).clamp(-12, 12)

    def decode(self, z):
        return self.decoder(self.expand(z).view(-1, self.base * 8, self.size // 16, self.size // 16))

    def forward(self, x):
        mu, logvar = self.encode(x)
        # DEMO P4-A: 先得到均值和方差，再加入标准正态噪声生成潜在编码 z。
        # exp(0.5*logvar) 得到标准差；训练时采样仍能传梯度，评估时用均值使重建稳定。
        z = mu + torch.exp(.5 * logvar) * torch.randn_like(mu) if self.training else mu
        return self.decode(z), mu, logvar


class DoubleConv(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False), nn.BatchNorm2d(outgoing), nn.ReLU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False), nn.BatchNorm2d(outgoing), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, base=32, classes=4):
        super().__init__()
        self.down = nn.ModuleList([DoubleConv(1, base), DoubleConv(base, base * 2),
                                  DoubleConv(base * 2, base * 4), DoubleConv(base * 4, base * 8)])
        self.bottleneck = DoubleConv(base * 8, base * 16)
        self.up = nn.ModuleList([DoubleConv(base * 24, base * 8), DoubleConv(base * 12, base * 4),
                                DoubleConv(base * 6, base * 2), DoubleConv(base * 3, base)])
        self.output = nn.Conv2d(base, classes, 1)

    def forward(self, x):
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for block, skip in zip(self.up, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            # DEMO P4-B: 把编码器保存的细节与解码器当前特征沿通道方向拼接。
            # 这样恢复图像尺寸时仍能利用边界细节；这里是拼接，不是残差相加。
            x = block(torch.cat([skip, x], dim=1))
        return self.output(x)  # Four categorical logits, not a single grayscale regression output.


class Generator(nn.Module):
    def __init__(self, latent=100, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent, base * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(base * 8), nn.ReLU(),
            nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(base * 4), nn.ReLU(),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(base * 2), nn.ReLU(),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1, bias=False), nn.BatchNorm2d(base), nn.ReLU(),
            nn.ConvTranspose2d(base, 1, 4, 2, 1), nn.Tanh())

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base, 4, 2, 1), nn.LeakyReLU(.2),
            nn.Conv2d(base, base * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(base * 2), nn.LeakyReLU(.2),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(base * 4), nn.LeakyReLU(.2),
            nn.Conv2d(base * 4, base * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(base * 8), nn.LeakyReLU(.2),
            nn.Conv2d(base * 8, 1, 4))

    def forward(self, x):
        return self.net(x).flatten()


def init_gan(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight, 0., .02)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1., .02)
            nn.init.zeros_(module.bias)
