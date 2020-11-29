import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.utils as vutils
import torch.nn.functional as F
import torch.nn.init as init

# This code is used for images with a size of 64*64


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

        

class  Attention_Layer(nn.Module):
    def __init__(self, ):
        super(Attention_Layer, self).__init__()


    def forward(self, x, w, bias, gamma):
        out = x.contiguous().view(x.size(0)*x.size(1), x.size(2))

        out_f = F.linear(out, w, bias)

        out = out_f.view(x.size(0),x.size(1), out_f.size(1))

        out= torch.sqrt((out**2).sum(2))

        alpha_01 = out /out.sum(1, keepdim=True).expand_as(out)

        alpha_01 = F.relu(alpha_01- 0.1/float(gamma))
        
        alpha_01 = alpha_01/alpha_01.sum(1, keepdim=True).expand_as(alpha_01)

        alpha = torch.unsqueeze(alpha_01, dim=2)
        out = alpha.expand_as(x)*x 

        return out, out_f, alpha_01




class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_units=16, num_class=10,image_channels=3):
        super(ResNet, self).__init__()
        self.num_units = num_units
        self.num_class = num_class
        self.in_planes = 64
        self.num_units_last=128

        self.expansion = block.expansion
    
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1)

        if self.expansion>1:
            self.conv4 = nn.Conv2d(512*self.expansion, 512, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn4 = nn.BatchNorm2d(512) 

        self.conv5 = nn.Conv2d(512, 512, kernel_size=9, stride=4, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn6 = nn.BatchNorm2d(512)

        self.attention_layer = Attention_Layer()

        self.linear  = nn.Linear(512, num_class)



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

        gamma = out.size(2)*out.size(3)

        if self.expansion>1:
            out = F.relu(self.bn4(self.conv4(out)))   
        
        out = F.relu(self.bn5(self.conv5(out)))
        out = F.relu(self.bn6(self.conv6(out)))

        out = out.view(out.size(0), out.size(1),-1)

        out = out.transpose(2,1)
        
        out, f, alpha = self.attention_layer(out, self.linear.weight, self.linear.bias, gamma)
       
        out = out.transpose(2,1).sum(2)

        out = out.view(out.size(0),-1)


        out = self.linear(out)
                
        return out, f, alpha


def ResNet18(num_class=10, image_channels=3):
    return ResNet(BasicBlock, [2,2,2,2], num_class=num_class, image_channels=image_channels)

def ResNet50(num_class=10, image_channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_class=num_class, image_channels=image_channels)
