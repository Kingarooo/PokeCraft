import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g, num_classes, img_size, embed_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.gen = nn.Sequential(
            self._block(z_dim+embed_size, features_g*16, 4, 1, 0),  # Entrada: Vetor de ruído
            self._block(features_g*16, features_g*8, 4, 2, 1),
            self._block(features_g*8, features_g*4, 4, 2, 1),
            self._block(features_g*4, features_g*2, 4, 2, 1),

            nn.ConvTranspose2d(features_g*2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Saída entre -1 e 1
        )
        self.embed = nn.Embedding(num_classes, embed_size)
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)

        x = torch.cat([x, embedding], dim=1)
        return self.gen(x)