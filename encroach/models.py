"""Multi-Task Deep Learning Architectures for Encroachment and Building Detection.

Uses segmentation_models_pytorch (SMP) as backbone with customized input channels (C=7)
and multi-task prediction heads:
1. Classification Head: binary building probability (logits)
2. Coverage Regression Head: continuous pixel coverage fraction in [0, 1]
3. Boundary Edge Head: building edge detection to maintain sharp footprint separation
"""
import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False


class FallbackDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class FallbackUNet(nn.Module):
    """Clean lightweight pure PyTorch UNet fallback when SMP is not yet installed."""
    def __init__(self, in_channels=7):
        super().__init__()
        self.inc = FallbackDoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), FallbackDoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), FallbackDoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), FallbackDoubleConv(256, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = FallbackDoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = FallbackDoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = FallbackDoubleConv(128, 64)
        
        # Multi-task heads
        self.head_mask = nn.Conv2d(64, 1, 1)
        self.head_cov = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())
        self.head_bnd = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        d3 = self.conv1(torch.cat([self.up1(x4), x3], dim=1))
        d2 = self.conv2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.conv3(torch.cat([self.up3(d2), x1], dim=1))
        
        return {
            "mask": self.head_mask(d1),
            "cov": self.head_cov(d1),
            "bnd": self.head_bnd(d1),
        }


class MultiTaskSMP(nn.Module):
    """Multi-task U-Net or UNet++ wrapping SMP with 7-channel input and 3 task heads."""
    def __init__(self, arch="Unet", encoder_name="resnet34", in_channels=7, encoder_weights="imagenet"):
        super().__init__()
        arch_cls = getattr(smp, arch, smp.Unet)
        # Base model with 1 output class
        self.base = arch_cls(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=64,  # output feature map dimension
            activation=None,
        )
        # 3 heads over base decoder features
        self.head_mask = nn.Conv2d(64, 1, kernel_size=1)
        self.head_cov = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1), nn.Sigmoid())
        self.head_bnd = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        feats = self.base(x)
        return {
            "mask": self.head_mask(feats),
            "cov": self.head_cov(feats),
            "bnd": self.head_bnd(feats),
        }


def build_model(cfg):
    """Instantiate model according to config settings."""
    if HAS_SMP:
        print(f"Building SMP {cfg.ARCH} model (encoder: {cfg.ENCODER}, in_channels=7)")
        return MultiTaskSMP(
            arch=cfg.ARCH,
            encoder_name=cfg.ENCODER,
            in_channels=7,
            encoder_weights=cfg.ENCODER_WEIGHTS,
        )
    else:
        print("SMP not installed; falling back to Pure PyTorch MultiTask UNet")
        return FallbackUNet(in_channels=7)
