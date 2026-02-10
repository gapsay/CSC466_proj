import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, padding=0),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, 3, padding=0),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bridge = DoubleConv(512, 1024)

        # Decoder 
        self.up0 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv0 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.dec0 = DoubleConv(1024, 512)

        self.up1 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.dec1 = DoubleConv(512, 256)

        self.up2 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec2 = DoubleConv(256, 128)

        self.up3 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec3 = DoubleConv(128, 64)

        self.out_layer = nn.Conv2d(64, out_channels, 1)

        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x):
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bridge(self.pool(e4))

        # Decode
        d0 = self.up_conv0(self.up0(b))
        d0 = torch.cat([d0, e4], dim=1)
        d0 = self.dec0(d0)

        d1 = self.up_conv1(self.up1(d0))
        d1 = torch.cat([d1, e3], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up_conv2(self.up2(d1))
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d3 = self.up_conv3(self.up3(d2))
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.dec3(d3)

        residual = torch.tanh(self.out_layer(d3))
        return x + residual