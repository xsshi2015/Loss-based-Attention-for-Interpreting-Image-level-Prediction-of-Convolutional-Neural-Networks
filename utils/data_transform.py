from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data


class DataTransform(data.Dataset):
    def __init__(self, trainData=None, trainLabel=None, trainIndex= None, train_patchLabel = None,train_boundingbox=None,
                 transform=None, target_transform=None,
                 download=False):
 
        self.train_data = trainData
        self.train_labels = trainLabel
        self.train_index = trainIndex
        self.transform = transform
        self.train_patchLabels = train_patchLabel

        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train_patchLabels is not None:
            img, target, patch_labels = self.train_data[index], self.train_labels[index], self.train_patchLabels[index]
        else:
            img, target = self.train_data[index], self.train_labels[index]


        img = np.array(img)
        img = np.transpose(img, (1,2,0))
        
        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        if self.train_patchLabels is not None:
            
            return img, target, patch_labels
        else:
            return img, target



    def __len__(self):
        return len(self.train_data)

    
    