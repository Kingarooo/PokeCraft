import torch
import torch.nn as nn
class Discriminator(nn.Module):
    def __init__(self, channels_img):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            self._block(channels_img, 64, 4, 2, 1),
            self._block(64, 128, 4, 2, 1),
            self._block(128, 256, 4, 2, 1),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()  # Saída entre 0 e 1
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.net(x)