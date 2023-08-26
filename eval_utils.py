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


def create_dataset(features, labels, data_is_numpy=True):
    if data_is_numpy:
        features = torch.from_numpy(features).type(torch.FloatTensor)
        labels = torch.from_numpy(labels).type(torch.LongTensor)
    return torch.utils.data.TensorDataset(features, labels)

def prepare_data_for_attack_evaluation(train_classifier_data, train_classifier_labels, test_data, test_labels, data_is_numpy):
    n_eval = min(len(train_classifier_data), len(test_data))    
    shuffle_idx = np.arange(n_eval)
    np.random.shuffle(shuffle_idx)
    known_idx = shuffle_idx[:n_eval // 2]
    unknown_idx = shuffle_idx[n_eval // 2:]

    # known train data
    known_train_features, known_train_labels = train_classifier_data[known_idx], train_classifier_labels[known_idx]
    known_train_data = create_dataset(known_train_features, known_train_labels, data_is_numpy=data_is_numpy)

    # unknown train data
    unknown_train_features, unknown_train_labels = train_classifier_data[unknown_idx], train_classifier_labels[unknown_idx]
    unknown_train_data = create_dataset(unknown_train_features, unknown_train_labels, data_is_numpy=data_is_numpy)

    # get non-member data
    known_test_features, known_test_labels = test_data[known_idx], test_labels[known_idx]
    unknown_test_features, unknown_test_labels = test_data[unknown_idx], test_labels[unknown_idx]

    # known test data
    known_test_data = create_dataset(known_test_features, known_test_labels, data_is_numpy=data_is_numpy)

    # unknown test data
    unknown_test_data = create_dataset(unknown_test_features, unknown_test_labels, data_is_numpy=data_is_numpy)

    return known_train_data, unknown_train_data, known_test_data, unknown_test_data


def evaluation_metrics(net, train_data, train_labels, test_data, test_labels, data_is_numpy=True):
    if data_is_numpy:
        num_classes = int(np.max(train_labels) + 1)
#         train_data_tensor = torch.from_numpy(train_data).type(torch.FloatTensor).cuda()
#         train_label_tensor = torch.from_numpy(train_labels).type(torch.LongTensor).cuda()
#         test_data_tensor = torch.from_numpy(test_data).type(torch.FloatTensor).cuda()
#         test_label_tensor = torch.from_numpy(test_labels).type(torch.LongTensor).cuda()
    else:
        num_classes = int(np.max(train_labels.numpy()) + 1)
#         train_data_tensor = train_data.type(torch.FloatTensor).cuda()
#         train_label_tensor = train_labels.type(torch.LongTensor).cuda()
#         test_data_tensor = test_data.type(torch.FloatTensor).cuda()
#         test_label_tensor = test_labels.type(torch.LongTensor).cuda()
    
#     train_outputs = net(train_data_tensor)
#     test_outputs = net(test_data_tensor)
#     train_acc = torch.sum(torch.argmax(
#         train_outputs, axis=1) == train_label_tensor.cuda()).item() / train_label_tensor.shape[0]
#     test_acc = torch.sum(torch.argmax(
#         test_outputs, axis=1) == test_label_tensor.cuda()).item() / test_label_tensor.shape[0]
    
    known_train_data, unknown_train_data, known_test_data, unknown_test_data = prepare_data_for_attack_evaluation(
        train_data, train_labels, test_data, test_labels, data_is_numpy=data_is_numpy)
    
    known_train_performance, known_test_performance, unknown_train_performance, unknown_test_performance = \
        prepare_model_performance(net, known_train_data, known_test_data,
                                  net, unknown_train_data, unknown_test_data)
    
    MIA = black_box_benchmarks(known_train_performance,
                               known_test_performance,
                               unknown_train_performance,
                               unknown_test_performance,
                               num_classes=num_classes)

    correctness_acc, confidence_acc, entropy_acc, mod_entropy_acc = MIA._mem_inf_benchmarks()    
#     gap_att_acc = 1/2 + (train_acc - test_acc) / 2
    return correctness_acc, confidence_acc, entropy_acc, mod_entropy_acc


def softmax_by_row(logits, T=1.0):
    mx = np.max(logits, axis=-1, keepdims=True)
    exp = np.exp((logits - mx) / T)
    denominator = np.sum(exp, axis=-1, keepdims=True)
    return exp / denominator

def prepare_model_performance(shadow_model, shadow_train_loader, shadow_test_loader,
                              target_model, target_train_loader,
                              target_test_loader, batch_size=256, device="cuda"):
    def _model_predictions(model, data):
        return_outputs, return_labels = [], []
        dataloader = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=True)
        for (inputs, labels) in dataloader:
            return_labels.append(labels.numpy())
            outputs = model(inputs.to(device))
            return_outputs.append(softmax_by_row(outputs.data.cpu().numpy()))
        return_outputs = np.concatenate(return_outputs)
        return_labels = np.concatenate(return_labels).astype(int)
        return (return_outputs, return_labels)

    shadow_train_performance = _model_predictions(shadow_model, shadow_train_loader)
    shadow_test_performance = _model_predictions(shadow_model, shadow_test_loader)

    target_train_performance = _model_predictions(target_model, target_train_loader)
    target_test_performance = _model_predictions(target_model, target_test_loader)
    return shadow_train_performance, shadow_test_performance, target_train_performance, target_test_performance


class black_box_benchmarks(object):

    def __init__(self, shadow_train_performance, shadow_test_performance,
                 target_train_performance, target_test_performance, num_classes, verbose=False):
        '''
        each input contains both model predictions (shape: num_data*num_classes) and ground-truth labels.
        '''
        self.num_classes = num_classes
        self.verbose = verbose

        self.s_tr_outputs, self.s_tr_labels = shadow_train_performance
        self.s_te_outputs, self.s_te_labels = shadow_test_performance
        self.t_tr_outputs, self.t_tr_labels = target_train_performance
        self.t_te_outputs, self.t_te_labels = target_test_performance

        self.s_tr_corr = (np.argmax(self.s_tr_outputs,
                                    axis=1) == self.s_tr_labels).astype(int)
        self.s_te_corr = (np.argmax(self.s_te_outputs,
                                    axis=1) == self.s_te_labels).astype(int)
        self.t_tr_corr = (np.argmax(self.t_tr_outputs,
                                    axis=1) == self.t_tr_labels).astype(int)
        self.t_te_corr = (np.argmax(self.t_te_outputs,
                                    axis=1) == self.t_te_labels).astype(int)
        self.s_tr_conf = np.array(
            [self.s_tr_outputs[i, self.s_tr_labels[i]] for i in
             range(len(self.s_tr_labels))])
        self.s_te_conf = np.array(
            [self.s_te_outputs[i, self.s_te_labels[i]] for i in
             range(len(self.s_te_labels))])
        self.t_tr_conf = np.array(
            [self.t_tr_outputs[i, self.t_tr_labels[i]] for i in
             range(len(self.t_tr_labels))])
        self.t_te_conf = np.array(
            [self.t_te_outputs[i, self.t_te_labels[i]] for i in
             range(len(self.t_te_labels))])

        self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        self.s_te_entr = self._entr_comp(self.s_te_outputs)
        self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        self.t_te_entr = self._entr_comp(self.t_te_outputs)

        self.s_tr_m_entr = self._m_entr_comp(self.s_tr_outputs, self.s_tr_labels)
        self.s_te_m_entr = self._m_entr_comp(self.s_te_outputs, self.s_te_labels)
        self.t_tr_m_entr = self._m_entr_comp(self.t_tr_outputs, self.t_tr_labels)
        self.t_te_m_entr = self._m_entr_comp(self.t_te_outputs, self.t_te_labels)

    def _log_value(self, probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))

    def _entr_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)), axis=1)

    def _m_entr_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1 - probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[
            range(true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[
            range(true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)

    def _thre_setting(self, tr_values, te_values):
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values >= value) / (len(tr_values) + 0.0)
            te_ratio = np.sum(te_values < value) / (len(te_values) + 0.0)
            acc = 0.5 * (tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre

    def _mem_inf_via_corr(self):
        # perform membership inference attack based on whether the input is correctly classified or not
        t_tr_acc = np.sum(self.t_tr_corr) / (len(self.t_tr_corr) + 0.0)
        t_te_acc = np.sum(self.t_te_corr) / (len(self.t_te_corr) + 0.0)
        mem_inf_acc = 0.5 * (t_tr_acc + 1 - t_te_acc)
        if self.verbose:
            print(
                'For membership inference attack via correctness, the attack acc is {acc1:.3f}, with train acc {acc2:.3f} and test acc {acc3:.3f}'.format(
                    acc1=mem_inf_acc, acc2=t_tr_acc, acc3=t_te_acc))
        return mem_inf_acc

    def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, t_tr_values,
                      t_te_values):
        # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
        # (negative) prediction entropy, and (negative) modified entropy
        t_tr_mem, t_te_non_mem = 0, 0
        for num in range(self.num_classes):
            thre = self._thre_setting(s_tr_values[self.s_tr_labels == num],
                                      s_te_values[self.s_te_labels == num])
            t_tr_mem += np.sum(t_tr_values[self.t_tr_labels == num] >= thre)
            t_te_non_mem += np.sum(t_te_values[self.t_te_labels == num] < thre)
        mem_inf_acc = 0.5 * (
                    t_tr_mem / (len(self.t_tr_labels) + 0.0) + t_te_non_mem / (
                        len(self.t_te_labels) + 0.0))
        if self.verbose:
            print(
                'For membership inference attack via {n}, the attack acc is {acc:.3f}'.format(
                    n=v_name, acc=mem_inf_acc))
        return mem_inf_acc

    def _mem_inf_benchmarks(self, all_methods=True, benchmark_methods=[]):
        correctness_acc, confidence_acc, entropy_acc, mod_entropy_acc = \
            None, None, None, None

        if (all_methods) or ('correctness' in benchmark_methods):
            correctness_acc = self._mem_inf_via_corr()
        if (all_methods) or ('confidence' in benchmark_methods):
            confidence_acc = self._mem_inf_thre('confidence', self.s_tr_conf, self.s_te_conf,
                               self.t_tr_conf, self.t_te_conf)
        if (all_methods) or ('entropy' in benchmark_methods):
            entropy_acc = self._mem_inf_thre('entropy', -self.s_tr_entr, -self.s_te_entr,
                               -self.t_tr_entr, -self.t_te_entr)
        if (all_methods) or ('modified entropy' in benchmark_methods):
            mod_entropy_acc = self._mem_inf_thre('modified entropy', -self.s_tr_m_entr,
                               -self.s_te_m_entr, -self.t_tr_m_entr,
                               -self.t_te_m_entr)

        return correctness_acc, confidence_acc, entropy_acc, mod_entropy_acc