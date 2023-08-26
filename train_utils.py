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
from torchvision.datasets import CIFAR100
# import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import tarfile
from sklearn.cluster import KMeans
import urllib
from copy import deepcopy
from torchvision.models import resnet18, resnet50


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_checkpoint(state, checkpoint, filename='checkpoint.pth.tar', filename_end='', is_best=False):
    filepath = os.path.join(checkpoint, filename)
    if not os.path.isdir(filepath):
        mkdir_p(filepath)
    filepath_full = os.path.join(filepath, filename_end)
    
    torch.save(state, filepath_full)
    if is_best:
        shutil.copyfile(filepath_full, os.path.join(checkpoint, 'model_best.pth.tar'))
        

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    

def load_data(dataset, load_randomization, use_validation=True, 
              train_classifier_ratio=0.1, train_attack_ratio=0.15, train_valid_ratio=0.25):
    
    if dataset == "texas":
        DATASET_NAME = f"{dataset}/100/feats"
    else:
        DATASET_NAME = f"dataset_{dataset}"
    DATASET_PATH = f"./datasets/{dataset}"

    if not os.path.isdir(DATASET_PATH):
        os.makedirs(DATASET_PATH)

    if dataset == "cifar":
        stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))

        train_dataset = CIFAR100(DATASET_PATH, train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ]))
        train_dataset_tensor = torch.stack([data for data, label in train_dataset])
        train_label_tensor = torch.Tensor([label for data, label in train_dataset])

        train_classifier_data = train_dataset_tensor[:20000]
        train_classifier_label = train_label_tensor[:20000]
        train_attack_data = train_dataset_tensor[20000:40000]
        train_attack_label = train_label_tensor[20000:40000]
        valid_data = train_dataset_tensor[40000:]
        valid_label = train_label_tensor[40000:]

        test_dataset = CIFAR100(DATASET_PATH, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ]))
        test_data = torch.stack([data for data, label in test_dataset])
        test_label = torch.Tensor([label for data, label in test_dataset])

    elif dataset in ["purchase", "texas"]:
        DATASET_NUMPY = 'data.npz'
        DATASET_FILE = os.path.join(DATASET_PATH, DATASET_NAME)

        if not os.path.isfile(DATASET_FILE):
            print('Dowloading the dataset...')
            urllib.request.urlretrieve(
                f"https://www.comp.nus.edu.sg/~reza/files/dataset_{dataset}.tgz",
                os.path.join(DATASET_PATH, 'tmp.tgz'))
            print('Dataset Dowloaded')
            tar = tarfile.open(os.path.join(DATASET_PATH, 'tmp.tgz'))
            tar.extractall(path=DATASET_PATH)

            if dataset == "purchase":
                data_set = np.genfromtxt(DATASET_FILE, delimiter=',')
                features = data_set[:, 1:].astype(np.float64)
                labels = (data_set[:, 0]).astype(np.int32) - 1
            elif dataset == "texas":
                dataset_features = np.genfromtxt(DATASET_FILE, delimiter=',')
                dataset_labels = os.path.join(DATASET_PATH, "texas/100/labels")
                dataset_labels = np.genfromtxt(dataset_labels, delimiter=',')
                features = dataset_features.astype(np.float64)
                labels = dataset_labels.astype(np.int32) - 1
            np.savez(os.path.join(DATASET_PATH, DATASET_NUMPY), X=features, Y=labels)

        data = np.load(os.path.join(DATASET_PATH, DATASET_NUMPY))
        X = data["X"]
        Y = data["Y"]

        len_train =len(X)

        if load_randomization:
            r = np.load(f"./datasets/{dataset}/random_r_{dataset}100.npy")
        else:
            r = np.arange(len_train)
            np.random.shuffle(r)

        X = X[r]
        Y = Y[r]

        # training data
        train_classifier_data = X[:int(train_classifier_ratio*len_train)]
        train_classifier_label = Y[:int(train_classifier_ratio*len_train)]

        # attack data
        train_attack_data = X[int(train_classifier_ratio*len_train):int((train_classifier_ratio+train_attack_ratio)*len_train)]
        train_attack_label = Y[int(train_classifier_ratio*len_train):int((train_classifier_ratio+train_attack_ratio)*len_train)]

        if use_validation:
            # validation data
            valid_data = X[int((train_classifier_ratio+train_attack_ratio)*len_train):
                int((train_classifier_ratio+train_attack_ratio+train_valid_ratio)*len_train)]
            valid_label = Y[int((train_classifier_ratio+train_attack_ratio)*len_train):
                int((train_classifier_ratio+train_attack_ratio+train_valid_ratio)*len_train)]

            # test data
            test_data = X[int((train_classifier_ratio+train_attack_ratio+train_valid_ratio)*len_train):]
            test_label = Y[int((train_classifier_ratio+train_attack_ratio+train_valid_ratio)*len_train):]
        else:
            # validation data
            valid_data = None
            valid_label = None

            # test data
            test_data = X[int((train_classifier_ratio+train_attack_ratio)*len_train):]
            test_label = Y[int((train_classifier_ratio+train_attack_ratio)*len_train):]

    return train_classifier_data, train_classifier_label, train_attack_data, train_attack_label, valid_data, valid_label, test_data, test_label


        
