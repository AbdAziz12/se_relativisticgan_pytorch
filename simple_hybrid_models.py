import torch
import torch.nn as nn

# ==========================================================
# Hybrid Generator:
# - Encoder enc1–enc3  : conv biasa
# - Encoder enc4–enc5  : separable
# - Decoder dec5–dec3  : deconv biasa
# - Decoder dec2–dec1  : separable deconv
# ==========================================================
class SimpleGenerator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, base_filters=8):
        super(SimpleGenerator, self).__init__()

        # ---------------- ENCODER ----------------
        self.enc1 = self._conv(input_channels, base_filters, 15, 2)
        self.enc2 = self._conv(base_filters, base_filters * 2, 15, 2)
        self.enc3 = self._conv(base_filters * 2, base_filters * 4, 15, 2)

        self.enc4 = self._separable_conv(base_filters * 4, base_filters * 8, 15, 2)
        self.enc5 = self._separable_conv(base_filters * 8, base_filters * 16, 15, 2)

        # Bottleneck
        self.bottleneck = self._separable_conv(
            base_filters * 16, base_filters * 16, 15, 1
        )

        # ---------------- DECODER ----------------
        # dec5–dec3 : separable upsample nearest
        self.dec5 = self._separable_upsample(base_filters * 16, base_filters * 8, 15)
        self.dec4 = self._separable_upsample(base_filters * 16, base_filters * 4, 15)
        # self.dec3 = self._separable_upsample(base_filters * 8, base_filters * 2, 15)

        # dec3–dec1 : upsample nearest + conv biasa
        self.dec3 = self._upsample_conv(base_filters * 8, base_filters * 2, 15)
        self.dec2 = self._upsample_conv(base_filters * 4, base_filters, 15)
        self.dec1 = self._upsample_conv(base_filters * 2, base_filters, 15)

        # # ---------------- DECODER ----------------
        # # deconv biasa (lebih kuat untuk rekonstruksi waveform)
        # self.dec5 = self._deconv(base_filters * 16, base_filters * 8, 15, 2)
        # self.dec4 = self._deconv(base_filters * 16, base_filters * 4, 15, 2)
        # self.dec3 = self._deconv(base_filters * 8, base_filters * 2, 15, 2)

        # # deconv separable (hemat resource)
        # self.dec2 = self._separable_deconv(base_filters * 4, base_filters, 15, 2)
        # self.dec1 = self._separable_deconv(base_filters * 2, base_filters, 15, 2)

        self.output = nn.Conv1d(base_filters, output_channels, 1)

    # ---------------- BASIC CONV ----------------
    def _conv(self, in_c, out_c, k, s):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, k, s, k // 2),
            nn.PReLU()
        )
    
    # def _deconv(self, in_c, out_c, k, s):
    #     return nn.Sequential(
    #         nn.ConvTranspose1d(in_c, out_c, k, s, k // 2, s - 1),
    #         nn.PReLU()
    #     )
    
    # def _separable_deconv(self, in_c, out_c, k, s):
    #     return nn.Sequential(
    #         nn.ConvTranspose1d(in_c, in_c, k, s, k // 2, s - 1, groups=in_c),
    #         nn.Conv1d(in_c, out_c, 1),
    #         nn.PReLU()
    #     )

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

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        b = self.bottleneck(e5)
        b = b + e5

        d5 = self.dec5(b)
        d4 = self.dec4(torch.cat([d5, e4], dim=1))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        return self.output(d1)


# ==========================================================
# Hybrid Discriminator:
# - conv1–conv3 : conv biasa
# - conv4–conv5 : separable
# ==========================================================
class SimpleDiscriminator(nn.Module):
    def __init__(self, input_channels=2, base_filters=8):
        super(SimpleDiscriminator, self).__init__()

        self.conv1 = self._conv(input_channels, base_filters, 15, 2)
        self.conv2 = self._conv(base_filters, base_filters * 2, 15, 2)
        self.conv3 = self._conv(base_filters * 2, base_filters * 4, 15, 2)

        self.conv4 = self._separable_conv(base_filters * 4, base_filters * 8, 15, 2)
        self.conv5 = self._separable_conv(base_filters * 8, base_filters * 16, 15, 2)

        self.output = nn.Conv1d(base_filters * 16, 1, 1)

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
        x = self.output(x)
        return torch.mean(x, dim=2)
