import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.utils as vutils
import torch.nn.functional as F
import torch.nn.init as init




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 =nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride !=1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
                )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck,self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)    
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)    
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride !=1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out += self.shortcut(x)
        out = F.relu(out)

        return out

        
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
        alpha02 = torch.squeeze(F.relu(alpha01.sum(1)-0.1/float(gamma)))
        alpha02 = alpha02/alpha02.sum(1, keepdim=True).expand_as(alpha02)

        alpha03 = torch.unsqueeze(alpha02, dim=1).expand_as(alpha01)
        alpha03 = alpha03.contiguous().view(x.size(0), 64*gamma)

        alpha[alpha03==0] = 0

        alpha = alpha/alpha.sum(1, keepdim=True).expand_as(alpha)

        alpha = torch.unsqueeze(alpha, dim=2)
        out = alpha.expand_as(x)*x 

        return out, torch.squeeze(out_f.sum(1)).view(x.size(0)*gamma, out_1.size(2)), alpha02



class Capsule(nn.Module):
    def __init__(self, block, num_blocks, num_units=16, num_class=10,image_channels=3):
        super(Capsule, self).__init__()
        self.num_units = num_units
        self.num_class = num_class
        self.in_planes = 64
        self.num_units_last=128
    
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1)
        
        self.linear  = nn.Linear(self.num_units_last, num_class)

        self.units = nn.ModuleList([ConvUnit(in_channels=512) for i in range(self.num_units)])

        self.b4 = nn.Parameter(torch.randn(1, 32,1,1))
        
        self.layer5 = nn.ModuleList([ConvUnit_small(32, 64, 1) for i in range(self.num_units_last)])

        self.attention_layer = Attention_Layer()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides =[stride]+[1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes *block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

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


def ResNet18(num_units=16, num_class=10, image_channels=3):
    return Capsule(BasicBlock, [2,2,2,2], num_units=16, num_class=num_class, image_channels=image_channels)