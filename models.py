from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import torch.nn.functional as F
import torch
import pickle
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import tarfile
from sklearn.cluster import KMeans
import urllib
from copy import deepcopy
from torchvision.models import resnet18, resnet50


class TabularClassifier(nn.Module):
    def __init__(self,num_classes=100, num_features=600):
        super(TabularClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(num_features,1024),
            nn.Tanh(),
            nn.Linear(1024,512),
            nn.Tanh(),
            nn.Linear(512,256),
            nn.Tanh(),
            nn.Linear(256,128),
            nn.Tanh(),
        )
        
        self.classifier = nn.Linear(128,num_classes)
        
    def forward(self,x):
        hidden_out = self.features(x)
        return self.classifier(hidden_out)
    

class InferenceAttack_HZ(nn.Module):
    def __init__(self,num_classes):
        self.num_classes=num_classes
        super(InferenceAttack_HZ, self).__init__()
        self.features=nn.Sequential(
            nn.Linear(100,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.ReLU(),
            )
        
        self.labels=nn.Sequential(
           nn.Linear(num_classes,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            )
        
        self.combine=nn.Sequential(
            nn.Linear(64*2,512),
            
            nn.ReLU(),
            nn.Linear(512,256),
            
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1),
            )
        
#         for key in self.state_dict():
#             print (key)
#             if key.split('.')[-1] == 'weight':    
#                 nn.init.normal(self.state_dict()[key], std=0.01)
#                 print (key)
                
#             elif key.split('.')[-1] == 'bias':
#                 self.state_dict()[key][...] = 0
        
        self.output= nn.Sigmoid()
        
    def forward(self,x, l):
        out_x = self.features(x)
        out_l = self.labels(l)
        is_member = self.combine(torch.cat((out_x,out_l), 1))
        return self.output(is_member)
