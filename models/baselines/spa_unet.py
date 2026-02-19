import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SPABlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(4)

        self.conv = nn.Conv2d(in_ch * 3, in_ch, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h, w = x.shape[2:]

        p1 = F.interpolate(self.pool1(x), size=(h, w), mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.pool2(x), size=(h, w), mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.pool3(x), size=(h, w), mode='bilinear', align_corners=False)

        out = torch.cat([p1, p2, p3], dim=1)
        attention = self.sigmoid(self.conv(out))

        return x * attention

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        return x1, x2, x3, x4

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class SPAUNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.encoder = Encoder()
        self.spa = SPABlock(512)

        self.up3 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.up1 = UpBlock(128, 64)

        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, pre, post):
        p1, p2, p3, p4 = self.encoder(pre)
        q1, q2, q3, q4 = self.encoder(post)

        f1 = torch.abs(p1 - q1)
        f2 = torch.abs(p2 - q2)
        f3 = torch.abs(p3 - q3)
        f4 = torch.abs(p4 - q4)

        bottleneck = self.spa(f4)

        d3 = self.up3(bottleneck, f3)
        d2 = self.up2(d3, f2)
        d1 = self.up1(d2, f1)

        return self.final(d1)
