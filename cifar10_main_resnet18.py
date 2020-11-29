#
# Dynamic Routing Between Capsules
# https://arxiv.org/pdf/1710.09829.pdf
#

from __future__ import print_function

import torch
import shutil
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import os
import argparse
import numpy as np
from model.conv_resnet import ResNet18
from loss.weight_loss import CrossEntropyLoss as CE



parser = argparse.ArgumentParser(description='Pytorch CIFAR10 Training')
parser.add_argument('-lr','--learning_rate', default='0.1', type=float, help='learning rate')
parser.add_argument('-r','--resume', action='store_true',help='resume from checkpoint')
parser.add_argument('-ch', '--checkpoint', metavar='DIR', help='path to checkpoint (default: ./checkpoint)', default='./checkpoint')



def save_checkpoint(state, is_best, prefix='', filename='./checkpoint/checkpoint.pt'):
    print("====> saving the new best model")
    torch.save(state, filename)
    if is_best:
        path = "/".join(filename.split('/')[:-1])
        best_filename = os.path.join(path, prefix+'_model_best'+'.pt')
        shutil.copyfile(filename, best_filename)




args = parser.parse_args()

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)


batch_size=128
num_class=10
image_channels=3
best_acc = 0
# Stop training if loss goes below this threshold.
early_stop_loss = 0.0001

# Normalization for CIFAR10 dataset.
normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

print('==> Preparing data..')
transforms_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
    ])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    normalize
    ])

train_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transforms_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.CIFAR10('../data', train=False, download=True, transform=transforms_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



arch = 'cifar10_conv_resnet18'
filename = arch +'_'+str(num_class)
checkpoint_filename = os.path.join(args.checkpoint, filename+'.pt')

    
net = ResNet18()

optimizer = torch.optim.SGD(net.parameters(), lr = 0.1, momentum=0.9, weight_decay=5e-4)
lr_update = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)


criterion = torch.nn.CrossEntropyLoss(size_average=True)
weight_criterion = CE(aggregate='sum')


use_cuda = torch.cuda.is_available()
if use_cuda:
    net.cuda()
    net=torch.nn.DataParallel(net,device_ids=range(torch.cuda.device_count()))
    criterion.cuda()
    weight_criterion.cuda()
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(123)



def rampup(global_step, rampup_length=80):
    if global_step <rampup_length:
        global_step = np.float(global_step)
        rampup_length = np.float(rampup_length)
        phase = 1.0 - np.maximum(0.0, global_step) / rampup_length
    else:
        phase = 0.0
    return np.exp(-5.0 * phase * phase)


def train(epoch):
    lr_update.step()

    net.train()
    train_loss = 0
    correct = 0
    total = 0

    rampup_value = rampup(epoch)

    if epoch==0:
        u_w = 0
    else:
        u_w = 0.1*rampup_value


    u_w = torch.autograd.Variable(torch.FloatTensor([u_w]).cuda(), requires_grad=False)   

    
    for batch_idx, data in enumerate(train_loader):
        inputs, targets = data

        if use_cuda:
            inputs, targets =inputs.cuda(), targets.cuda()

        optimizer.zero_grad()

        inputs, targets = Variable(inputs), Variable(targets)

        outputs, out_f, alpha = net(inputs)

        loss_1 = criterion(outputs, targets)

        loss_2 = weight_criterion(out_f, targets.repeat(6*6,1).permute(1,0).contiguous().view(-1), weights=alpha.view(-1))
        
        loss = loss_1 + u_w*loss_2/outputs.size(0)

        loss.backward()
        optimizer.step()


        
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets).cpu().sum()

        #print("Training accuracy is:{}".format(100.*correct/total))
    print("Epoch[{}]: Loss: {:.4f} Train accuracy: {}".format(epoch, loss.data[0], 100.*correct/total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0.0
    total = 0.0

    for batch_idx, (inputs, targets) in enumerate(test_loader):

        with torch.no_grad():
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets =Variable(inputs), Variable(targets)
            outputs,_,_ = net(inputs)

            _, predicted = torch.max(outputs.data,1)
            total +=targets.size(0)
            correct +=predicted.eq(targets.data).cpu().sum()

    print("Epoch[{}] Test accuracy: {}".format(epoch, float(100.*correct)/total))


    # Save checkpoint.
    acc =100.*float(correct)/total
    if acc > best_acc:
        save_checkpoint({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'best_prec' : best_acc,
                }, is_best=1, prefix=arch, filename=checkpoint_filename)

        best_acc = acc
    print("Best accuracy is:{}".format(best_acc))





num_epochs = 200
for epoch in range(0, num_epochs):
    train(epoch)
    test(epoch)
