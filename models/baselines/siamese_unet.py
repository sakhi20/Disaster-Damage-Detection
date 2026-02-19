import torch
import torch.nn as nn

class ConvBlock(nn.Module):
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

class SiameseUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Shared encoder
        self.enc1 = ConvBlock(3, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)

        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up2 = nn.ConvTranspose2d(512, 128, 2, stride=2)
        self.dec2 = ConvBlock(384, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(192, 64)

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, pre, post):
        # Encoder (shared)
        p1 = self.enc1(pre)
        p2 = self.enc2(self.pool(p1))
        p3 = self.enc3(self.pool(p2))

        q1 = self.enc1(post)
        q2 = self.enc2(self.pool(q1))
        q3 = self.enc3(self.pool(q2))

        # Feature fusion
        f3 = torch.cat([p3, q3], dim=1)

        # Decoder
        d2 = self.up2(f3)
        d2 = self.dec2(torch.cat([d2, p2, q2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, p1, q1], dim=1))

        return torch.sigmoid(self.final(d1))
