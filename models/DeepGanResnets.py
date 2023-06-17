import torch
import torch.nn as nn

class ResblockUP(nn.Module):
    def __init__(self, in_channels, channelreducefactor=2, neuter=False, scale_factor=2):
        super().__init__()
        if neuter:
            self.upsample = nn.Identity()
            channelreducefactor=1
        else:
          self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.channelreducefactor = channelreducefactor
        self.network = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels//4, kernel_size=(1,1), bias=False),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(),
            self.upsample, 
            nn.Conv2d(in_channels//4, in_channels//4, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(),
            nn.Conv2d(in_channels//4, in_channels//4, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(),
            nn.Conv2d(in_channels//4, in_channels//channelreducefactor, kernel_size=(1,1), bias=False)
        )
    def forward(self, xb):
        return self.network(xb) + self.upsample(xb[:,0:xb.shape[1]//self.channelreducefactor,:,:])
    
class ResblockDown(nn.Module):
    def __init__(self, in_channels, channelincreasefactor=2, neuter=False):
        super().__init__()
        if neuter:
            self.pool = nn.Identity()
            channelincreasefactor=1
        else:
          self.pool = nn.AvgPool2d(2,2)
        self.network = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels,in_channels//2, kernel_size=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, in_channels//2, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, in_channels//2, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            self.pool,
            nn.Conv2d(in_channels//2, in_channels*channelincreasefactor, kernel_size=(1,1))
        )
        if channelincreasefactor == 1:
            self.extra_conv = nn.Identity()
        else:
          self.extra_conv = nn.Conv2d(in_channels, in_channels*(channelincreasefactor-1), kernel_size=(1,1))
        self.incf = channelincreasefactor
        self.neuter = neuter
    def forward(self, xb):
        output = self.network(xb)
        if not self.neuter:
          xb = self.pool(xb)
        if self.incf > 1:
          xb = torch.cat((xb, self.extra_conv(xb)), dim=1)
        return output + xb