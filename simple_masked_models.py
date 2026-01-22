import torch
import torch.nn as nn

# ==========================================================
# Hybrid Generator:
# ==========================================================
class SimpleGenerator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, base_filters=8):
        super(SimpleGenerator, self).__init__()

        # # ---------------- ENCODER V4 ----------------
        # self.enc1 = self._conv(input_channels, base_filters, 15, 2)
        # self.enc2 = self._conv(base_filters, base_filters * 2, 15, 2)
        # self.enc3 = self._conv(base_filters * 2, base_filters * 4, 15, 2)

        # self.enc4 = self._separable_conv(base_filters * 4, base_filters * 8, 15, 2)
        # self.enc5 = self._separable_conv(base_filters * 8, base_filters * 16, 15, 2)

        # # Bottleneck
        # self.bottleneck = self._res_block(base_filters * 16, 7)

        # # ---------------- DECODER V4 ----------------
        # self.dec5 = self._separable_deconv(base_filters * 16, base_filters * 8, 15, 2)
        # self.dec4 = self._separable_deconv(base_filters * 16, base_filters * 4, 15, 2)

        # self.dec3 = self._deconv(base_filters * 8, base_filters * 2, 15, 2)
        # self.dec2 = self._deconv(base_filters * 4, base_filters, 15, 2)
        # self.dec1 = self._upsample_conv(base_filters * 2, base_filters, 15)

        # # ---------------- ENCODER V6 Base 12 ----------------
        # self.enc1 = self._conv(input_channels, base_filters, 15, 2)
        # self.enc2 = self._conv(base_filters, base_filters * 2, 15, 2)
        # self.enc3 = self._conv(base_filters * 2, base_filters * 2, 15, 2)

        # self.enc4 = self._conv(base_filters * 2, base_filters * 4, 15, 2)
        # self.enc5 = self._conv(base_filters * 4, base_filters * 8, 15, 2)

        # # Bottleneck
        # self.bottleneck = self._res_block(base_filters * 8, 7)

        # # ---------------- DECODER V6 Base 12 ----------------
        # self.dec5 = self._deconv(base_filters * 8, base_filters * 4, 15, 2)
        # self.dec4 = self._deconv(base_filters * 8, base_filters * 2, 15, 2)

        # self.dec3 = self._deconv(base_filters * 4, base_filters * 2, 15, 2)
        # self.dec2 = self._deconv(base_filters * 4, base_filters, 15, 2)
        # # self.dec1 = self._upsample_conv(base_filters * 2, base_filters, 15)
        # self.dec1 = self._deconv(base_filters * 2, base_filters, 15, 2)

        # # ---------------- ENCODER V7 Base 10 ----------------
        # self.enc1 = self._conv(input_channels, base_filters, 15, 2)
        # self.enc2 = self._conv(base_filters, base_filters * 2, 15, 2)
        # self.enc3 = self._conv(base_filters * 2, base_filters * 2, 15, 2)

        # self.enc4 = self._conv(base_filters * 2, base_filters * 4, 15, 2)
        # self.enc5 = self._conv(base_filters * 4, base_filters * 8, 15, 2)
        # self.enc6 = self._separable_conv(base_filters * 8, base_filters * 8, 15, 2)

        # # Bottleneck
        # self.bottleneck = self._res_block(base_filters * 8, 7)

        # # ---------------- DECODER V7 Base 10 ----------------
        # self.dec6 = self._separable_deconv(base_filters * 8, base_filters * 8, 15, 2)
        # self.dec5 = self._deconv(base_filters * 16, base_filters * 4, 15, 2)
        # self.dec4 = self._deconv(base_filters * 8, base_filters * 2, 15, 2)

        # self.dec3 = self._deconv(base_filters * 4, base_filters * 2, 15, 2)
        # self.dec2 = self._deconv(base_filters * 4, base_filters, 15, 2)
        # self.dec1 = self._upsample_conv(base_filters * 2, base_filters, 15)
        # # self.dec1 = self._deconv(base_filters * 2, base_filters, 15, 2)

        # # ---------------- ENCODER V8 Base 8 ----------------
        # self.enc1 = self._conv(input_channels, base_filters, 15, 2)
        # self.enc2 = self._conv(base_filters, base_filters * 2, 15, 2)
        # self.enc3 = self._conv(base_filters * 2, base_filters * 2, 15, 2)

        # self.enc4 = self._conv(base_filters * 2, base_filters * 4, 15, 2)
        # self.enc5 = self._conv(base_filters * 4, base_filters * 8, 15, 2)
        # self.enc6 = self._conv(base_filters * 8, base_filters * 8, 15, 2)

        # # Bottleneck
        # self.bottleneck = self._res_block(base_filters * 8, 7)

        # # ---------------- DECODER V8 Base 8 ----------------
        # self.dec6 = self._deconv(base_filters * 8, base_filters * 8, 15, 2)
        # self.dec5 = self._deconv(base_filters * 16, base_filters * 4, 15, 2)
        # self.dec4 = self._deconv(base_filters * 8, base_filters * 2, 15, 2)

        # self.dec3 = self._deconv(base_filters * 4, base_filters * 2, 15, 2)
        # self.dec2 = self._deconv(base_filters * 4, base_filters, 15, 2)
        # self.dec1 = self._upsample_conv(base_filters * 2, base_filters, 15)
        # # self.dec1 = self._deconv(base_filters * 2, base_filters, 15, 2)

        # ---------------- ENCODER V8 Base 10 ----------------
        self.enc1 = self._conv(input_channels, base_filters, 15, 2)
        self.enc2 = self._conv(base_filters, base_filters * 2, 15, 2)
        self.enc3 = self._conv(base_filters * 2, base_filters * 2, 15, 2)

        self.enc4 = self._conv(base_filters * 2, base_filters * 4, 15, 2)
        self.enc5 = self._conv(base_filters * 4, base_filters * 8, 15, 2)

        # Bottleneck
        self.bottleneck = self._res_block(base_filters * 8, 7, 2)

        # ---------------- DECODER V8 Base 10 ----------------
        self.dec5 = self._deconv(base_filters * 8, base_filters * 4, 15, 2)
        self.dec4 = self._deconv(base_filters * 8, base_filters * 2, 15, 2)

        self.dec3 = self._deconv(base_filters * 4, base_filters * 2, 15, 2)
        self.dec2 = self._deconv(base_filters * 4, base_filters, 15, 2)
        self.dec1 = self._upsample_conv(base_filters * 2, base_filters, 15)
        # self.dec1 = self._deconv(base_filters * 2, base_filters, 15, 2)

        self.output = nn.Conv1d(base_filters, output_channels, 1)
        # self.output = self._output_mask(base_filters, output_channels, 1)

    # ---------------- BASIC CONV ----------------
    def _conv(self, in_c, out_c, k, s):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, k, s, k // 2),
            nn.PReLU()
        )
    
    def _deconv(self, in_c, out_c, k, s):
        return nn.Sequential(
            nn.ConvTranspose1d(in_c, out_c, k, s, k // 2, s - 1),
            nn.PReLU()
        )
    
    def _separable_deconv(self, in_c, out_c, k, s):
        return nn.Sequential(
            nn.ConvTranspose1d(in_c, in_c, k, s, k // 2, s - 1, groups=in_c),
            nn.Conv1d(in_c, out_c, 1),
            nn.PReLU()
        )

    # ---------------- UPSAMPLE BLOCKS ----------------
    def _upsample_conv(self, in_c, out_c, k):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(in_c, out_c, k, stride=1, padding=k // 2),
            nn.PReLU()
        )

    def _separable_upsample(self, in_c, out_c, k):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(in_c, in_c, k, stride=1, padding=k // 2, groups=in_c),
            nn.Conv1d(in_c, out_c, 1),
            nn.PReLU()
        )

    # ---------------- SEPARABLE CONV ----------------
    def _separable_conv(self, in_c, out_c, k, s):
        return nn.Sequential(
            nn.Conv1d(in_c, in_c, k, s, k // 2, groups=in_c),
            nn.Conv1d(in_c, out_c, 1),
            nn.PReLU()
        )
    
    def _res_block(self, in_c, k, d=1):
        return nn.Sequential(
            nn.Conv1d(in_c, in_c * 2, 1),       # Expansion: Perbanyak fitur
            nn.PReLU(),
            nn.Conv1d(in_c * 2, in_c * 2, k, 1, d * (k // 2), dilation=d, groups=in_c * 2), # Depthwise
            nn.PReLU(),
            nn.Conv1d(in_c * 2, in_c, 1),       # Projection: Kembalikan ukuran
        )
    
    def _output_mask(self, in_c, out_c, k):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, k),
            # nn.Sigmoid()
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        b = self.bottleneck(e5)
        b = b + e5
        d5 = self.dec5(b)

        # # 6 Layer
        # e6 = self.enc6(e5)
        # b = self.bottleneck(e6)
        # b = b + e6
        # d6 = self.dec6(b)
        # d5 = self.dec5(torch.cat([d6, e5], dim=1))

        d4 = self.dec4(torch.cat([d5, e4], dim=1))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        mask = self.output(d1)

        # return self.output(d1)
        return mask + x
        # return mask

# ==========================================================
# Hybrid Discriminator:
# ==========================================================
class SimpleDiscriminator(nn.Module):
    def __init__(self, input_channels=2, base_filters=8):
        super(SimpleDiscriminator, self).__init__()

        # # V4
        # self.conv1 = self._conv(input_channels, base_filters, 15, 2)
        # self.conv2 = self._conv(base_filters, base_filters * 2, 15, 2)
        # self.conv3 = self._conv(base_filters * 2, base_filters * 4, 15, 2)

        # self.conv4 = self._separable_conv(base_filters * 4, base_filters * 8, 15, 2)
        # self.conv5 = self._separable_conv(base_filters * 8, base_filters * 16, 15, 2)

        # self.output = nn.Conv1d(base_filters * 16, 1, 1)

        # V4
        self.conv1 = self._conv(input_channels, base_filters, 15, 2)
        self.conv2 = self._conv(base_filters, base_filters * 2, 15, 2)
        self.conv3 = self._conv(base_filters * 2, base_filters * 2, 15, 2)

        self.conv4 = self._conv(base_filters * 2, base_filters * 4, 15, 2)
        self.conv5 = self._conv(base_filters * 4, base_filters * 8, 15, 2)

        self.output = nn.Conv1d(base_filters * 8, 1, 1)

        # # V4
        # self.conv1 = self._conv(input_channels, base_filters, 15, 2)
        # self.conv2 = self._conv(base_filters, base_filters * 2, 15, 2)
        # self.conv3 = self._conv(base_filters * 2, base_filters * 2, 15, 2)

        # self.conv4 = self._conv(base_filters * 2, base_filters * 4, 15, 2)
        # self.conv5 = self._conv(base_filters * 4, base_filters * 8, 15, 2)
        # self.conv6 = self._separable_conv(base_filters * 8, base_filters * 8, 15, 2)

        # self.output = nn.Conv1d(base_filters * 8, 1, 1)

    # ---- Utilities ----
    def _conv(self, in_c, out_c, k, s):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, k, s, k // 2),
            nn.LeakyReLU(0.25)
        )

    def _separable_conv(self, in_c, out_c, k, s):
        return nn.Sequential(
            nn.Conv1d(in_c, in_c, k, s, k // 2, groups=in_c),
            nn.Conv1d(in_c, out_c, 1),
            nn.LeakyReLU(0.25)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.conv6(x)
        x = self.output(x)

        return torch.mean(x, dim=2)
