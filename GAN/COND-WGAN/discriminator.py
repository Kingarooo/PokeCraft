import torch
import torch.nn as nn
class Discriminator(nn.Module):
    def __init__(self, channels_img,features_d, num_classes, img_size):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.disc = nn.Sequential(
            #Input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img+1, features_d, kernel_size=4, stride=2, padding=1), #32x32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1),
            self._block(features_d*2, features_d*4, 4, 2, 1),
            self._block(features_d*4, features_d*8, 4, 2, 1),
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),
        )
        self.embed = nn.Embedding(num_classes, img_size*img_size)

    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True), #Layer norm <-> Instance norm
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], -1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1) # N*C*img_size*img_size
        return self.disc(x)