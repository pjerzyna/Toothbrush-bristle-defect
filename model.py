import torch
import torch.nn as nn

class UNet(nn.Module):
    """
    Standard U-Net architecture for binary image segmentation.
    Uses 3 downsampling/upsampling blocks with skip connections.
    """
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = self.block(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = self.block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = self.block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self.block(256, 512)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.block(128, 64)

        # Output layer (1 channel for binary mask)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def block(self, in_c: int, out_c: int) -> nn.Sequential:
        """Returns a standard convolutional block (Conv -> BN -> ReLU x2)."""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder path with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out(d1))