import torch
import torch.nn as nn

class SimpleGenerator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, base_filters=8): # Base filter turun ke 8
        super(SimpleGenerator, self).__init__()

        # # Encoder (Hybrid: Layer 1 Conv Biasa, Sisanya Separable)
        # self.enc1 = nn.Sequential(nn.Conv1d(input_channels, base_filters, 15, 2, 7), nn.PReLU()) # 4096
        # self.enc2 = self._separable_conv(base_filters, base_filters * 2, 15, 2)
        # self.enc3 = self._separable_conv(base_filters * 2, base_filters * 2, 15, 2)
        # self.enc4 = self._separable_conv(base_filters * 2, base_filters * 4, 15, 2)
        # self.enc5 = self._separable_conv(base_filters * 4, base_filters * 8, 15, 2)
        
        # self.bottleneck = self._separable_conv(base_filters * 8, base_filters * 8, 15, 1) # 128

        # # Decoder (Separable Transpose)
        # self.dec5 = self._separable_deconv(base_filters * 8, base_filters * 4, 15, 2)
        # self.dec4 = self._separable_deconv(base_filters * 8, base_filters * 2, 15, 2) # In: d5+e4 = 32+32=64
        # self.dec3 = self._separable_deconv(base_filters * 4, base_filters * 2, 15, 2) # In: d4+e3 = 16+16=32
        # self.dec2 = self._separable_deconv(base_filters * 4, base_filters, 15, 2)     # In: d3+e2 = 16+16=32
        # self.dec1 = self._separable_deconv(base_filters * 2, base_filters, 15, 2)     # In: d2+e1 = 8+8=16
        
        # self.output = nn.Conv1d(base_filters, output_channels, 1)

        # ENCODER 
        # Layer 1: 1 -> 8
        self.enc1 = nn.Sequential(nn.Conv1d(input_channels, base_filters, 15, 2, 7), nn.PReLU())
        # Layer 2: 8 -> 16
        self.enc2 = self._separable_conv(base_filters, base_filters * 2, 15, 2)
        # Layer 3: 16 -> 32 (Perubahan di sini)
        self.enc3 = self._separable_conv(base_filters * 2, base_filters * 4, 15, 2)
        # Layer 4: 32 -> 64
        self.enc4 = self._separable_conv(base_filters * 4, base_filters * 8, 15, 2)
        # Layer 5: 64 -> 128
        self.enc5 = self._separable_conv(base_filters * 8, base_filters * 16, 15, 2)
        
        # Bottleneck tetapkan di 128 ch
        self.bottleneck = self._separable_conv(base_filters * 16, base_filters * 16, 15, 1)

        # DECODER
        # dec5: 128 -> 64
        self.dec5 = self._separable_deconv(base_filters * 16, base_filters * 8, 15, 2)
        
        # dec4: (dec5:64 + enc4:64) = 128 -> 32
        self.dec4 = self._separable_deconv(base_filters * 16, base_filters * 4, 15, 2)
        
        # dec3: (dec4:32 + enc3:32) = 64 -> 16
        self.dec3 = self._separable_deconv(base_filters * 8, base_filters * 2, 15, 2)
        
        # dec2: (dec3:16 + enc2:16) = 32 -> 8
        self.dec2 = self._separable_deconv(base_filters * 4, base_filters, 15, 2)
        
        # dec1: (dec2:8 + enc1:8) = 16 -> 8
        self.dec1 = self._separable_deconv(base_filters * 2, base_filters, 15, 2)
        
        self.output = nn.Conv1d(base_filters, output_channels, 1)
        
    def _separable_conv(self, in_c, out_c, k, s):
        return nn.Sequential(
            nn.Conv1d(in_c, in_c, k, s, k//2, groups=in_c), # Depthwise
            nn.Conv1d(in_c, out_c, 1),                     # Pointwise
            nn.PReLU()
        )
    
    def _separable_deconv(self, in_c, out_c, k, s):
        return nn.Sequential(
            nn.ConvTranspose1d(in_c, in_c, k, s, k//2, s-1, groups=in_c), # Depthwise Transpose
            nn.Conv1d(in_c, out_c, 1),                                   # Pointwise
            nn.PReLU()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        b = self.bottleneck(e5)
        b = (b + e5) * 0.5 # Residual Connection
        
        d5 = self.dec5(b)
        d4 = self.dec4(torch.cat([d5, e4], 1))
        d3 = self.dec3(torch.cat([d4, e3], 1))
        d2 = self.dec2(torch.cat([d3, e2], 1))
        d1 = self.dec1(torch.cat([d2, e1], 1))
        
        return self.output(d1)

class SimpleDiscriminator(nn.Module):
    def __init__(self, input_channels=2, base_filters=8):
        super(SimpleDiscriminator, self).__init__()
        
        # Layer awal biasa, tengah separable
        self.conv1 = nn.Sequential(nn.Conv1d(input_channels, base_filters, 15, 2, 7), nn.LeakyReLU(0.25))
        self.conv2 = self._separable_conv(base_filters, base_filters * 2, 15, 2)
        self.conv3 = self._separable_conv(base_filters * 2, base_filters * 4, 15, 2)
        self.conv4 = self._separable_conv(base_filters * 4, base_filters * 8, 15, 2)
        self.conv5 = self._separable_conv(base_filters * 8, base_filters * 16, 15, 2)
        
        self.output = nn.Conv1d(base_filters * 16, 1, 1)
        
    def _separable_conv(self, in_c, out_c, k, s):
        return nn.Sequential(
            nn.Conv1d(in_c, in_c, k, s, k//2, groups=in_c),
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