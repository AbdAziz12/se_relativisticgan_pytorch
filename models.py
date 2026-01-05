import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """
    Generator untuk SERGAN menggunakan arsitektur fully convolutional
    dengan encoder-decoder dan skip connections
    """
    def __init__(self, input_channels=1, output_channels=1, base_filters=16):
        super(Generator, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(input_channels, base_filters, kernel_size=31, stride=2)
        self.enc2 = self._conv_block(base_filters, base_filters * 2, kernel_size=31, stride=2)
        self.enc3 = self._conv_block(base_filters * 2, base_filters * 2, kernel_size=31, stride=2)
        self.enc4 = self._conv_block(base_filters * 2, base_filters * 4, kernel_size=31, stride=2)
        self.enc5 = self._conv_block(base_filters * 4, base_filters * 4, kernel_size=31, stride=2)
        self.enc6 = self._conv_block(base_filters * 4, base_filters * 8, kernel_size=31, stride=2)
        self.enc7 = self._conv_block(base_filters * 8, base_filters * 8, kernel_size=31, stride=2)
        self.enc8 = self._conv_block(base_filters * 8, base_filters * 16, kernel_size=31, stride=2)
        self.enc9 = self._conv_block(base_filters * 16, base_filters * 16, kernel_size=31, stride=2)
        self.enc10 = self._conv_block(base_filters * 16, base_filters * 32, kernel_size=31, stride=2)
        self.enc11 = self._conv_block(base_filters * 32, base_filters * 32, kernel_size=31, stride=2)
        
        # Decoder (dengan skip connections)
        self.dec11 = self._deconv_block(base_filters * 32, base_filters * 32, kernel_size=31, stride=2)
        self.dec10 = self._deconv_block(base_filters * 64, base_filters * 16, kernel_size=31, stride=2)
        self.dec9 = self._deconv_block(base_filters * 32, base_filters * 16, kernel_size=31, stride=2)
        self.dec8 = self._deconv_block(base_filters * 32, base_filters * 8, kernel_size=31, stride=2)
        self.dec7 = self._deconv_block(base_filters * 16, base_filters * 8, kernel_size=31, stride=2)
        self.dec6 = self._deconv_block(base_filters * 16, base_filters * 4, kernel_size=31, stride=2)
        self.dec5 = self._deconv_block(base_filters * 8, base_filters * 4, kernel_size=31, stride=2)
        self.dec4 = self._deconv_block(base_filters * 8, base_filters * 2, kernel_size=31, stride=2)
        self.dec3 = self._deconv_block(base_filters * 4, base_filters * 2, kernel_size=31, stride=2)
        self.dec2 = self._deconv_block(base_filters * 4, base_filters, kernel_size=31, stride=2)
        self.dec1 = self._deconv_block(base_filters * 2, base_filters, kernel_size=31, stride=2)
        
        # Output layer
        self.output = nn.Conv1d(base_filters, output_channels, kernel_size=1)
        # self.tanh = nn.Tanh()
        
    def _conv_block(self, in_channels, out_channels, kernel_size, stride):
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.PReLU()
        )
    
    def _deconv_block(self, in_channels, out_channels, kernel_size, stride):
        padding = kernel_size // 2
        output_padding = stride - 1
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, 
                             padding, output_padding),
            nn.PReLU()
        )
    
    def forward(self, x):
        # Encoder dengan skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)
        e9 = self.enc9(e8)
        e10 = self.enc10(e9)
        e11 = self.enc11(e10)
        
        # Decoder dengan skip connections
        d11 = self.dec11(e11)
        d10 = self.dec10(torch.cat([d11, e10], dim=1))
        d9 = self.dec9(torch.cat([d10, e9], dim=1))
        d8 = self.dec8(torch.cat([d9, e8], dim=1))
        d7 = self.dec7(torch.cat([d8, e7], dim=1))
        d6 = self.dec6(torch.cat([d7, e6], dim=1))
        d5 = self.dec5(torch.cat([d6, e5], dim=1))
        d4 = self.dec4(torch.cat([d5, e4], dim=1))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        
        # output = self.tanh(self.output(d1))
        output = self.output(d1)
        return output


class Discriminator(nn.Module):
    """
    Discriminator untuk SERGAN
    Input: concatenation dari clean/generated dan noisy audio
    """
    def __init__(self, input_channels=2, base_filters=16):
        super(Discriminator, self).__init__()
        
        self.conv1 = self._conv_block(input_channels, base_filters, kernel_size=31, stride=2)
        self.conv2 = self._conv_block(base_filters, base_filters * 2, kernel_size=31, stride=2)
        self.conv3 = self._conv_block(base_filters * 2, base_filters * 2, kernel_size=31, stride=2)
        self.conv4 = self._conv_block(base_filters * 2, base_filters * 4, kernel_size=31, stride=2)
        self.conv5 = self._conv_block(base_filters * 4, base_filters * 4, kernel_size=31, stride=2)
        self.conv6 = self._conv_block(base_filters * 4, base_filters * 8, kernel_size=31, stride=2)
        self.conv7 = self._conv_block(base_filters * 8, base_filters * 8, kernel_size=31, stride=2)
        self.conv8 = self._conv_block(base_filters * 8, base_filters * 16, kernel_size=31, stride=2)
        self.conv9 = self._conv_block(base_filters * 16, base_filters * 16, kernel_size=31, stride=2)
        self.conv10 = self._conv_block(base_filters * 16, base_filters * 32, kernel_size=31, stride=2)
        self.conv11 = self._conv_block(base_filters * 32, base_filters * 32, kernel_size=31, stride=2)
        
        # Output layer (tanpa sigmoid untuk WGAN/Relativistic GAN)
        self.output = nn.Conv1d(base_filters * 32, 1, kernel_size=1)
        
    def _conv_block(self, in_channels, out_channels, kernel_size, stride):
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            # nn.InstanceNorm1d(out_channels),  # Instance Normalization seperti paper
            nn.LeakyReLU(0.3)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.output(x)
        
        # Global average pooling untuk mendapatkan single value
        x = torch.mean(x, dim=2)
        return x


class SimpleGenerator(nn.Module):
    """
    Generator sederhana untuk testing dengan resource terbatas
    """
    def __init__(self, input_channels=1, output_channels=1, base_filters=16):
        super(SimpleGenerator, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(input_channels, base_filters, 31, 2)
        self.enc2 = self._conv_block(base_filters, base_filters * 2, 31, 2)
        self.enc3 = self._conv_block(base_filters * 2, base_filters * 4, 31, 2)
        self.enc4 = self._conv_block(base_filters * 4, base_filters * 8, 31, 2)
        self.enc5 = self._conv_block(base_filters * 8, base_filters * 16, 31, 2)
        
        # Decoder
        self.dec5 = self._deconv_block(base_filters * 16, base_filters * 8, 31, 2)
        self.dec4 = self._deconv_block(base_filters * 16, base_filters * 4, 31, 2)
        self.dec3 = self._deconv_block(base_filters * 8, base_filters * 2, 31, 2)
        self.dec2 = self._deconv_block(base_filters * 4, base_filters, 31, 2)
        self.dec1 = self._deconv_block(base_filters * 2, base_filters, 31, 2)
        
        self.output = nn.Conv1d(base_filters, output_channels, 1)
        self.tanh = nn.Tanh()
        
    def _conv_block(self, in_c, out_c, k, s):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, k, s, k//2),
            nn.PReLU()
        )
    
    def _deconv_block(self, in_c, out_c, k, s):
        return nn.Sequential(
            nn.ConvTranspose1d(in_c, out_c, k, s, k//2, s-1),
            nn.PReLU()
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        d5 = self.dec5(e5)
        d4 = self.dec4(torch.cat([d5, e4], 1))
        d3 = self.dec3(torch.cat([d4, e3], 1))
        d2 = self.dec2(torch.cat([d3, e2], 1))
        d1 = self.dec1(torch.cat([d2, e1], 1))
        
        return self.tanh(self.output(d1))


class SimpleDiscriminator(nn.Module):
    """
    Discriminator sederhana untuk testing
    """
    def __init__(self, input_channels=2, base_filters=16):
        super(SimpleDiscriminator, self).__init__()
        
        self.conv1 = self._conv_block(input_channels, base_filters, 31, 2)
        self.conv2 = self._conv_block(base_filters, base_filters * 2, 31, 2)
        self.conv3 = self._conv_block(base_filters * 2, base_filters * 4, 31, 2)
        self.conv4 = self._conv_block(base_filters * 4, base_filters * 8, 31, 2)
        self.conv5 = self._conv_block(base_filters * 8, base_filters * 16, 31, 2)
        
        self.output = nn.Conv1d(base_filters * 16, 1, 1)
        
    def _conv_block(self, in_c, out_c, k, s):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, k, s, k//2),
            nn.InstanceNorm1d(out_c),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.output(x)
        return torch.mean(x, dim=2)