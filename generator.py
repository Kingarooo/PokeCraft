import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim, channels_img):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(noise_dim, 256, 4, 1, 0),  # Entrada: Vetor de ruído
            self._block(256, 128, 4, 2, 1),
            self._block(128, 64, 4, 2, 1),
            nn.ConvTranspose2d(64, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Saída entre -1 e 1
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)