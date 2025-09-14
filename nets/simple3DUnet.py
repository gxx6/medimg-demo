import torch
import torch.nn as nn
import torch.nn.functional as F

class Double_conv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class simple_uet_model(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Encoder
        self.c1 = Double_conv(in_channels, 16, dropout_rate=0.1)
        self.p1 = nn.MaxPool3d(2)

        self.c2 = Double_conv(16, 32, dropout_rate=0.1)
        self.p2 = nn.MaxPool3d(2)

        self.c3 = Double_conv(32, 64, dropout_rate=0.2)
        self.p3 = nn.MaxPool3d(2)

        self.c4 = Double_conv(64, 128, dropout_rate=0.2)
        self.p4 = nn.MaxPool3d(2)

        self.c5 = Double_conv(128, 256, dropout_rate=0.3)

        # Decoder
        self.u6 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.c6 = Double_conv(128 + 128, 128, dropout_rate=0.2)

        self.u7 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.c7 = Double_conv(64 + 64, 64, dropout_rate=0.2)

        self.u8 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.c8 = Double_conv(32 + 32, 32, dropout_rate=0.1)

        self.u9 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.c9 = Double_conv(16 + 16, 16, dropout_rate=0.1)

        self.out_conv = nn.Conv3d(16, num_classes, kernel_size=1)

        self._init_weights_kaiming_uniform()

    def _init_weights_kaiming_uniform(self):
        for m in self.modules():
            if isinstance(m,(nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        c1 = self.c1(x); p1 = self.p1(c1)
        c2 = self.c2(p1); p2 = self.p2(c2)
        c3 = self.c3(p2); p3 = self.p3(c3)
        c4 = self.c4(p3); p4 = self.p4(c4)
        c5 = self.c5(p4)

        u6 = self.u6(c5); u6 = torch.cat([u6, c4], dim=1); c6 = self.c6(u6)
        u7 = self.u7(c6); u7 = torch.cat([u7, c3], dim=1); c7 = self.c7(u7)
        u8 = self.u8(c7); u8 = torch.cat([u8, c2], dim=1); c8 = self.c8(u8)
        u9 = self.u9(c8); u9 = torch.cat([u9, c1], dim=1); c9 = self.c9(u9)

        logits = self.out_conv(c9)   # (B, C, D, H, W)
        return logits                # ← 训练阶段返回 logits，不做 softmax
