import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.utils as vutils
import torch.nn.functional as F
import torch.nn.init as init

cfg={
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64,'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,512, 'M', 512, 512, 512, 512,'M'],
}

class ConvUnit(nn.Module):
    def __init__(self, in_channels):
        super(ConvUnit, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=in_channels,
                               out_channels=32,  # fixme constant
                               kernel_size=9,  # fixme constant
                               stride=4, # fixme constant
                               bias=True)
        self.bn= nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv0(x)))
        return out

class ConvUnit_small(nn.Module):
    def __init__(self, in_channels,out_channels, strides):
        super(ConvUnit_small, self).__init__()


        self.conv0 = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,  # fixme constant
                                kernel_size=1,  # fixme constant
                                stride=1, # fixme constant
                                bias=True)
        
        self.bn0= nn.BatchNorm2d(out_channels)



        self.shortcut = nn.Sequential()
        if strides != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides, bias=True), 
            )
    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)))

        return out

class  Attention_Layer(nn.Module):
    def __init__(self, ):
        super(Attention_Layer, self).__init__()


    def forward(self, x, w, bias, gamma):
        out = x.contiguous().view(x.size(0)*x.size(1), x.size(2))
        
        # out = F.relu(F.linear(out, w_0, bias_0))
        out = F.linear(out, w, bias)

        out_1 = out.view(x.size(0), x.size(1), out.size(1))

        out_f = out_1.view(x.size(0), 64, gamma, out_1.size(2))


        out= torch.sqrt((out_1**2).sum(2))

        alpha = out /out.sum(1, keepdim=True).expand_as(out)
 

        alpha01 = alpha.view(x.size(0), 64, gamma)
        alpha02 = torch.squeeze(F.relu(alpha01.sum(1)-0.1/float(gamma)))  # To achieve better localization performance, please use alpha02 = torch.squeeze(F.relu(alpha01.sum(1)-1.0/float(gamma)))
        alpha02 = alpha02/alpha02.sum(1, keepdim=True).expand_as(alpha02)

        alpha03 = torch.unsqueeze(alpha02, dim=1).expand_as(alpha01)
        alpha03 = alpha03.contiguous().view(x.size(0), 64*gamma)

        alpha[alpha03==0] = 0

        alpha = alpha/alpha.sum(1, keepdim=True).expand_as(alpha)

        alpha = torch.unsqueeze(alpha, dim=2)
        out = alpha.expand_as(x)*x 

        return out, torch.squeeze(out_f.sum(1)).view(x.size(0)*gamma, out_1.size(2)), alpha02



class Capsule(nn.Module):
    def __init__(self, vgg_name, num_units=16, num_class=10,image_channels=3):
        super(Capsule, self).__init__()
        self.num_units = num_units
        self.num_class = num_class
        self.in_planes = 64
        self.image_channels = image_channels

        self.num_units_last=128

        self.features = self._make_layers(cfg[vgg_name])

        # self.linear0 = nn.Linear(num_units, 512)
        self.linear  = nn.Linear(self.num_units_last, num_class)


        self.linear  = nn.Linear(self.num_units_last, num_class)

        self.units = nn.ModuleList([ConvUnit(in_channels=512) for i in range(self.num_units)])

        self.b4 = nn.Parameter(torch.randn(1, 32,1,1))
        
        self.layer5 = nn.ModuleList([ConvUnit_small(32, 64, 1) for i in range(self.num_units_last)])

        self.attention_layer = Attention_Layer()


    def _make_layers(self, cfg):
        layers = []
        in_channels = self.image_channels
        for x in cfg:
            if x is not 'M':
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                nn.BatchNorm2d(x),
                nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        
        # primary
        out = [self.units[i](out) for i in range(self.num_units)]
        out = torch.stack(out, dim=4)
        out = torch.sqrt((out**2).sum(dim=4))
        out = F.relu(out-self.b4.expand_as(out))

        gamma= out.size(2)*out.size(3)

        u = [self.layer5[i](out) for i in range(self.num_units_last)]
        u = torch.stack(u, dim=1)
        u = u.view(out.size(0), self.num_units_last, -1)
        z = u.transpose(1, 2)


        u_hat, u_f, alpha = self.attention_layer(z, self.linear.weight, self.linear.bias, gamma)
        
        f = torch.squeeze(u_hat.sum(1))

        y = self.linear(f)


        return y, u_f, alpha

    