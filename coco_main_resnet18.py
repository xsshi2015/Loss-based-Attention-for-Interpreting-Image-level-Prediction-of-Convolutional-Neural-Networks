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
from model.conv_resnet_stride_2 import ResNet18
from utils.calculate_AP import calculate_AP
from utils.data_transform import DataTransform as DT
from loss.weight_sigmoid_loss import SigmoidLoss as SL
import h5py
import hdf5storage


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

# Stop training if loss goes below this threshold.
early_stop_loss = 0.0001

# Normalization for CIFAR10 dataset.
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

print('==> Preparing data..')
transforms_train = transforms.Compose([
    transforms.RandomCrop(64, padding=4), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
    ])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    normalize
    ])


# load images
f = h5py.File('/data/.data5/xiaoshuang/Deep_Capsule_Net_Loss_Attention/Image_localization/data/coco_trainData_patchLabel_32.mat','r')
trainData = np.transpose(np.array(f['trainImages']),(3,2,1,0))



trainLabel = np.transpose(np.squeeze(f['trainLabels']),(1,0))
train_dataset = DT(trainData=trainData, trainLabel=trainLabel, transform=transforms_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


f = h5py.File('/data/.data5/xiaoshuang/Deep_Capsule_Net_Loss_Attention/Image_localization/data/coco_testData_patchLabel_32.mat','r')
n = len(f['testImages'][0])


testData = np.transpose(np.array(f['testImages']),(3,2,1,0))
testLabel = np.transpose(np.squeeze(f['testLabels']),(1,0))
testBoundingBox = np.transpose(np.squeeze(f['patchLabels']),(2,1,0))


test_dataset = DT(trainData=testData, trainLabel=testLabel, train_patchLabel = testBoundingBox, transform=transforms_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


num_class=91
image_channels=3
arch = 'coco_resnet18'
filename = arch +'_'+str(num_class)
checkpoint_filename = os.path.join(args.checkpoint, filename+'.pt')

    
net = ResNet18(num_class=num_class)
criterion = torch.nn.BCEWithLogitsLoss(size_average=False)
weight_criterion = SL(aggregate='sum')

use_cuda = torch.cuda.is_available()
if use_cuda:
    net.cuda()
    net=torch.nn.DataParallel(net,device_ids=range(torch.cuda.device_count()))
    criterion.cuda()
    weight_criterion.cuda()
    cudnn.benchmark = True



def rampup(global_step, rampup_length=80):
    if global_step <rampup_length:
        global_step = np.float(global_step)
        rampup_length = np.float(rampup_length)
        phase = 1.0 - np.maximum(0.0, global_step) / rampup_length
    else:
        phase = 0.0
    return np.exp(-5.0 * phase * phase)




optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)   # if criterion with size_average=True, try lr=0.1
lr_step = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)  

def train(epoch):
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

        if inputs.size(3)==3:
            inputs = inputs.permute(0,3,1,2)

        inputs = inputs.float()

        if use_cuda:
            inputs, targets =inputs.cuda(), targets.cuda()

        one_hot_target = torch.squeeze(targets.float())
        
        optimizer.zero_grad()

        inputs, one_hot_target = Variable(inputs), Variable(one_hot_target)

        outputs, out_f, alpha = net(inputs)

        loss_1 = criterion(outputs, one_hot_target)

        loss_2 = weight_criterion(out_f, one_hot_target.repeat(1,6*6).contiguous().view(out_f.size(0), out_f.size(1)), weights=alpha.view(-1))
        
        loss = loss_1/outputs.size(0) + u_w*loss_2/outputs.size(0)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()


        vector_labels = one_hot_target.data.byte().clone()  
        predicted = (outputs.data>0.5)
        total += vector_labels[vector_labels==1].sum()
        correct += ((vector_labels==predicted)[vector_labels==1]).sum()
        
        acc = 100.*correct/total
        
    lr_step.step()
    print("Epoch[{}]: Loss: {:.4f} Train accuracy: {}".format(epoch, loss.item(), 100.*correct/total))


best_acc=0
def ap_eval(epoch):
    global best_acc
    net.eval()
    prediction_weights = []
    true_labels = []

    for batch_idx, (inputs, targets, _) in enumerate(test_loader):

        if inputs.size(3)==3:
            inputs = inputs.permute(0,3,1,2)

        inputs = inputs.float()
        targets = torch.squeeze(targets.float())

        with torch.no_grad():
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets =Variable(inputs), Variable(targets)
            outputs, _,_ = net(inputs)

            vector_labels = targets.data.cpu().numpy() 
            predicted = outputs.data.cpu().numpy()

            prediction_weights.extend(predicted)
            
            true_labels.extend(vector_labels)
    
    prediction_weights = torch.from_numpy(np.array(prediction_weights))
    true_labels = torch.from_numpy(np.array(true_labels))


    AP = torch.squeeze(torch.zeros(num_class,1))

    for iter in range(num_class):
        AP[iter] = calculate_AP(prediction_weights[:,iter], true_labels[:,iter])

    # print("AP is:{}".format(AP))
    mAP = AP.mean()

    print("Epoch[{}], mAP is:{}".format(epoch, mAP))

    if mAP > best_acc:
        save_checkpoint({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'best_prec' : mAP,
                }, is_best=1, prefix=arch, filename=checkpoint_filename)

        best_acc = mAP

    print("Best accuracy is:{}".format(best_acc))



def patch_eval(epoch):
    
    best_filename = arch + '_model_best'
    best_checkpoint_filename = os.path.join(args.checkpoint, best_filename+'.pt')

    ##--- load from best checkpoint if exists ---
    if os.path.isfile(best_checkpoint_filename):
        print("=> loading checkpoint '{}'".format(best_checkpoint_filename))
        checkpoint = torch.load(best_checkpoint_filename)
        net.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(best_checkpoint_filename, checkpoint['epoch']))
    elif os.path.isfile(checkpoint_filename):
        print("=> loading checkpoint '{}'".format(checkpoint_filename))
        checkpoint = torch.load(checkpoint_filename)
        net.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_filename))
        return
    
    net.eval()

    prediction_weights = []
    true_labels = []
    true_image_labels = []
    weights =[]
    correct = 0 
    total = 0

    for batch_idx, (inputs, targets, patch_labels) in enumerate(test_loader):

        if inputs.size(3)==3:
            inputs = inputs.permute(0,3,1,2)

        inputs = inputs.float()
        # targets = torch.squeeze(targets.float())

        with torch.no_grad():
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets =Variable(inputs), Variable(targets)
            _, patch_outputs, alpha = net(inputs)

            predicted = patch_outputs.data.cpu().numpy()

            prediction_weights.extend(predicted)

            weights.extend(alpha.view(-1).data.cpu().numpy())
            
            true_labels.extend(patch_labels.numpy())
            
            true_image_labels.extend(targets.data.cpu().numpy())
            
    prediction_weights = torch.from_numpy(np.array(prediction_weights)).float()
    weights = torch.from_numpy(np.squeeze(np.array(weights))).float()
    true_labels = torch.from_numpy(np.array(true_labels)).float()
    true_image_labels_0 = torch.from_numpy(np.array(true_image_labels)).float()


    true_image_labels = true_image_labels_0.repeat(1,6*6).contiguous().view(prediction_weights.size(0), prediction_weights.size(1))
    
    _, predicted = torch.max(prediction_weights,1)


    true_image_labels = true_image_labels_0.repeat(1,6*6).contiguous().view(prediction_weights.size(0), prediction_weights.size(1))

    
    _, predicted = torch.max(prediction_weights,1)

    one_hot_predicted = []

    predicted = torch.sigmoid(prediction_weights)

    b_predicted = predicted>0.5

    one_hot_predicted = b_predicted.float()


    A = (one_hot_predicted*true_image_labels)>0
    A = A.float()

    A = A.sum(1)>0

    B_true_labels = true_labels.sum(2)>0

    total += B_true_labels.sum()
    correct += (A[weights>0][B_true_labels.view(-1)[weights>0]]).sum()

    recall  = float(100.*correct)/float(total)
    precision = float(100.*correct)/float(A[weights>0].sum())

    print("Epoch[{}] Test accuracy: {}/{}".format(epoch, precision, recall))


num_epochs = 200
for epoch in range(0, num_epochs):
    train(epoch)
    ap_eval(epoch)
    #patch_eval(epoch)
