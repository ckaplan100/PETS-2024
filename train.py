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

from models import *
from train_utils import *

def train(train_data, labels, model, criterion, optimizer, batch_size, epoch, device, num_batchs=999999, 
          add_grad_noise=False, grad_noise_scale=5, loss_weights=None, verbose=False, mmd_weight=None, mmd_scale=1,
          ref_data=None, ref_labels=None, start_mmd_epoch=2, unique_labels=False, mmd_ref_term=False):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    end = time.time()
    len_t =  (len(train_data)//batch_size)-1
    
    for ind in range(len_t):
        if ind > num_batchs:
            break
        inputs = train_data[ind*batch_size:(ind+1)*batch_size]        
        targets = labels[ind*batch_size:(ind+1)*batch_size]
        if loss_weights is not None:
            batch_loss_weights = loss_weights[ind*batch_size:(ind+1)*batch_size]

        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)

        # compute output
        outputs = model(inputs)
        
        if loss_weights is not None:
            loss = torch.mean(criterion(outputs, targets) * batch_loss_weights)
        else:
            loss = torch.mean(criterion(outputs, targets))
            
        if mmd_weight is not None and epoch > start_mmd_epoch:
            inputs_ref = ref_data[ind*batch_size:(ind+1)*batch_size].to(device)
            targets_ref = ref_labels[ind*batch_size:(ind+1)*batch_size]

            if unique_labels:
            # calculate mmd value unique per label
                unique_train_labels = torch.unique(targets)
                mmd_vals_per_label = torch.zeros(unique_train_labels.shape)
#                 skipped_labels = 0
                for label_idx, unique_label in enumerate(unique_train_labels):
                    n_instances_train = len(inputs[targets == unique_label.item()])
                    n_instances_ref = len(inputs_ref[targets_ref == unique_label.item()])
                    if n_instances_train < 2 or n_instances_ref < 2:
#                         skipped_labels += 1
                        continue
                    output_one_label = model(inputs[targets == unique_label.item()]).unsqueeze(0)
                    output_one_label_ref = model(inputs_ref[targets_ref == unique_label.item()]).unsqueeze(1)
                    label_train_term = torch.exp(-0.5*torch.norm(output_one_label - output_one_label, dim=-1)**2 / mmd_scale**2).mean()
                    label_ref_term = torch.exp(-0.5*torch.norm(output_one_label_ref - output_one_label_ref, dim=-1)**2 / mmd_scale**2).mean()
                    label_cross_term = -2 * torch.exp(-0.5*torch.norm(output_one_label - output_one_label_ref, dim=-1)**2 / mmd_scale**2).mean()
                    if mmd_ref_term:   
                        mmd_vals_per_label[label_idx] = label_cross_term + label_train_term + label_ref_term
                    else:
                        mmd_vals_per_label[label_idx] = label_cross_term + label_train_term
                    mmd_val = torch.mean(mmd_vals_per_label).to(device)
#                 print(f"Skipped label pct: {skipped_labels / len(unique_train_labels)}")
            else:    
                # calculate mmd value all together for batch
                outputs_ref = model(inputs_ref)
                outputs_mmd = outputs.unsqueeze(0)
                outputs_ref_mmd = outputs.unsqueeze(1)

                train_term = torch.exp(-0.5*torch.norm(outputs_mmd - outputs_mmd, dim=-1)**2 / mmd_scale**2).mean()
                ref_term = torch.exp(-0.5*torch.norm(outputs_ref_mmd - outputs_ref_mmd, dim=-1)**2 / mmd_scale**2).mean()
                cross_term = -2 * torch.exp(-0.5*torch.norm(outputs_mmd - outputs_ref_mmd, dim=-1)**2 / mmd_scale**2).mean()

                mmd_val = cross_term + train_term
            loss += (mmd_weight * mmd_val)
                
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data, inputs.size()[0])
        top1.update(prec1, inputs.size()[0])
        top5.update(prec5, inputs.size()[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
#         loss.backward(retain_graph=True)
#         erm_loss_grad = torch.norm(model.features[0].weight.grad)
#         if mmd_weight is not None and epoch > start_mmd_epoch:
#             mmd_val.backward()
#             mmd_loss_grad = torch.norm(model.features[0].weight.grad) - erm_loss_grad
#             print(erm_loss_grad, mmd_loss_grad)        
        
        # add noise to gradient
        if add_grad_noise:
            scale = 1 / grad_noise_scale
            for i in range(len(model.features)): 
                try:
                    grad_noise = draw_laplace_noise(model.features[i].weight.shape).to(device)
                    grad_noise *= scale
                    model.features[i].weight.grad += grad_noise
                except AttributeError:
                    continue
        
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if verbose and ind % 100 == 0:
            print('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=ind + 1,
                    size=len_t,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    ))

    return losses.avg, top1.avg


def test(test_data, labels, model, criterion, batch_size, epoch, device, verbose=False):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    len_t =  (len(test_data)//batch_size)-1
    
    for ind in range(len_t):
        # measure data loading time
        inputs = test_data[ind*batch_size:(ind+1)*batch_size]
        targets = labels[ind*batch_size:(ind+1)*batch_size]

        
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)

        # compute output
        outputs = model(inputs)
        
        loss = torch.mean(criterion(outputs, targets))

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data, inputs.size()[0])
        top1.update(prec1, inputs.size()[0])
        top5.update(prec5, inputs.size()[0])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if verbose and ind % 100 == 0:
            print('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=ind + 1,
                        size=len(test_data),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        ))

    return (losses.avg, top1.avg)


def train_privately(training_style, train_data, labels, model, inference_model, criterion, optimizer, batch_size, 
                   epoch, device, num_batchs=10000,skip_batch=0,alpha=0.5, 
                   attack_data=None, attack_labels=None,
                   i=None, squared_loss=False, log_loss=False, non_member_loss_term=False, verbose=False
                  ):

    # switch to train mode
    model.train()
    inference_model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    
    
    len_t =  (len(train_data)//batch_size)-1
    
    if training_style == "standard":
        for ind in range(len_t):
            inputs = train_data[ind*batch_size:(ind+1)*batch_size]
            targets = labels[ind*batch_size:(ind+1)*batch_size]

            data_time.update(time.time() - end)

            inputs, targets = inputs.to(device), targets.to(device)

            # compute output
            outputs = model(inputs)

            if device == "cuda":
                one_hot_tr = torch.from_numpy((np.zeros((outputs.size()[0],outputs.size(1))))).to(device).type(torch.cuda.FloatTensor)
                target_one_hot_tr = one_hot_tr.scatter_(1, targets.type(torch.cuda.LongTensor).view([-1,1]).data,1)
            else:
                one_hot_tr = torch.from_numpy((np.zeros((outputs.size()[0],outputs.size(1))))).to(device).type(torch.FloatTensor)
                target_one_hot_tr = one_hot_tr.scatter_(1, targets.type(torch.LongTensor).view([-1,1]).data,1)
            infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)

            inference_output = inference_model(outputs, infer_input_one_hot)

            if squared_loss is True and log_loss is True:
                raise ValueError("squared_loss and log_loss cannot both be used at the same time")
            
            if non_member_loss_term:
                inputs_att = attack_data[ind*batch_size:(ind+1)*batch_size]
                targets_att = attack_labels[ind*batch_size:(ind+1)*batch_size]
                inputs_att, targets_att = inputs_att.to(device), targets_att.to(device)
                outputs_att = model(inputs_att)

                if device == "cuda":
                    one_hot_att = torch.from_numpy((np.zeros((outputs_att.size()[0],outputs_att.size(1))))).to(device).type(torch.cuda.FloatTensor)
                    target_one_hot_att = one_hot_att.scatter_(1, targets_att.type(torch.cuda.LongTensor).view([-1,1]).data,1)
                else:
                    one_hot_att = torch.from_numpy((np.zeros((outputs_att.size()[0],outputs_att.size(1))))).to(device).type(torch.FloatTensor)
                    target_one_hot_att = one_hot_att.scatter_(1, targets_att.type(torch.LongTensor).view([-1,1]).data,1)      
                
                infer_input_one_hot_att = torch.autograd.Variable(target_one_hot_att)
                inference_output_att = inference_model(outputs_att, infer_input_one_hot_att)
                if squared_loss:
                    # best loss
                    loss = criterion(outputs, targets) + (alpha * torch.mean(torch.pow(inference_output, 2))) + (alpha * torch.mean(torch.pow(1 - inference_output_att, 2)))
                elif log_loss:
                    loss = criterion(outputs, targets) + (alpha * torch.mean(torch.log(inference_output))) + (alpha * torch.mean(torch.log(1 - inference_output_att)))
                else:
                    loss = criterion(outputs, targets) + alpha * torch.mean(inference_output) + alpha * torch.mean(1 - inference_output_att)
            else:
                if squared_loss:
                    loss = criterion(outputs, targets) + (alpha * torch.mean(torch.pow(inference_output, 2)))
                elif log_loss:
                    # paper loss
                    loss = criterion(outputs, targets) + (alpha * torch.mean(torch.log(inference_output)))
                else: 
                    # original code loss
                    loss = criterion(outputs, targets) + alpha * torch.mean(inference_output)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.data, inputs.size()[0])
            top1.update(prec1, inputs.size()[0])
            top5.update(prec5, inputs.size()[0])

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            if verbose and ind % 100 == 0:
                print(alpha, '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=ind + 1,
                        size=len_t,
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        ))
    
    elif training_style == "coin_flip":
        inputs = train_data[i*batch_size:(i+1)*batch_size]
        targets = labels[i*batch_size:(i+1)*batch_size]
        
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)

        # compute output
        outputs = model(inputs)

        if device == "cuda":
            one_hot_tr = torch.from_numpy((np.zeros((outputs.size()[0],outputs.size(1))))).to(device).type(torch.cuda.FloatTensor)
            target_one_hot_tr = one_hot_tr.scatter_(1, targets.type(torch.cuda.LongTensor).view([-1,1]).data,1)
        else:
            one_hot_tr = torch.from_numpy((np.zeros((outputs.size()[0],outputs.size(1))))).to(device).type(torch.FloatTensor)
            target_one_hot_tr = one_hot_tr.scatter_(1, targets.type(torch.LongTensor).view([-1,1]).data,1)
        
        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)

        inference_output = inference_model(outputs, infer_input_one_hot)

        relu = nn.ReLU()

        if non_member_loss_term:
            inputs_att = attack_data[i*batch_size:(i+1)*batch_size]
            targets_att = attack_labels[i*batch_size:(i+1)*batch_size]
            inputs_att, targets_att = inputs_att.to(device), targets_att.to(device)
            outputs_att = model(inputs_att)

            if device == "cuda":
                one_hot_att = torch.from_numpy((np.zeros((outputs_att.size()[0],outputs_att.size(1))))).to(device).type(torch.cuda.FloatTensor)
                target_one_hot_att = one_hot_att.scatter_(1, targets_att.type(torch.cuda.LongTensor).view([-1,1]).data,1)
            else:
                one_hot_att = torch.from_numpy((np.zeros((outputs_att.size()[0],outputs_att.size(1))))).to(device).type(torch.FloatTensor)
                target_one_hot_att = one_hot_att.scatter_(1, targets_att.type(torch.LongTensor).view([-1,1]).data,1)    
            
            infer_input_one_hot_att = torch.autograd.Variable(target_one_hot_att)
            inference_output_att = inference_model(outputs_att, infer_input_one_hot_att)
            if squared_loss:
                # best loss
                loss = criterion(outputs, targets) + (alpha * torch.mean(torch.pow(inference_output, 2))) + (alpha * torch.mean(torch.pow(1 - inference_output_att, 2)))
            elif log_loss:
                loss = criterion(outputs, targets) + (alpha * torch.mean(torch.log(inference_output))) + (alpha * torch.mean(torch.log(1 - inference_output_att)))
            else:
                loss = criterion(outputs, targets) + alpha * torch.mean(inference_output) + alpha * torch.mean(1 - inference_output_att)
        else:
            if squared_loss:
                loss = criterion(outputs, targets) + (alpha * torch.mean(torch.pow(inference_output, 2)))
            elif log_loss:
                # paper loss
                loss = criterion(outputs, targets) + (alpha * torch.mean(torch.log(inference_output)))
            else: 
                # original code loss
                loss = criterion(outputs, targets) + alpha * torch.mean(inference_output)

#         if i > 120:
#             print(criterion(outputs, targets))
#             print((alpha * torch.mean(torch.pow(inference_output, 2))))
#             if non_member_loss_term:
#                 print((alpha * torch.mean(torch.pow(1 - inference_output_att, 2))))
        

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data, inputs.size()[0])
        top1.update(prec1, inputs.size()[0])
        top5.update(prec5, inputs.size()[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if verbose and i % 100 == 0:
            print(alpha, '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=i + 1,
                    size=len_t,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    ))
        
    elif training_style == "code":
        for ind in range(skip_batch,len_t):

            if ind >= skip_batch+num_batchs:
                break
            
            # measure data loading time


            inputs = train_data[ind*batch_size:(ind+1)*batch_size]
            targets = labels[ind*batch_size:(ind+1)*batch_size]

            data_time.update(time.time() - end)

            inputs, targets = inputs.to(device), targets.to(device)

            # compute output
            outputs = model(inputs)

            one_hot_tr = torch.from_numpy((np.zeros((outputs.size()[0],outputs.size(1))))).to(device).type(torch.cuda.FloatTensor)
            target_one_hot_tr = one_hot_tr.scatter_(1, targets.type(torch.cuda.LongTensor).view([-1,1]).data,1)
            infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)

            inference_output = inference_model(outputs, infer_input_one_hot)

            relu = nn.ReLU()
            
            loss = criterion(outputs, targets) + alpha*torch.mean(inference_output)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.data, inputs.size()[0])
            top1.update(prec1, inputs.size()[0])
            top5.update(prec5, inputs.size()[0])

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            if verbose and ind%100==0:
                print(alpha, '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=ind + 1,
                        size=len_t,
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        ))
        
    else:
        raise ValueError("unhandled training style")

    return (losses.avg, top1.avg)
        
                
def train_attack(training_style, train_data, labels, attack_data, attack_label, model, attack_model, criterion, 
                 attack_criterion, optimizer, attack_optimizer, batch_size, epoch, device,
                 num_batchs=100000,skip_batch=0, i=None, verbose=False):
    # switch to train mode
    model.eval()
    attack_model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    end = time.time()
    len_t =  min((len(attack_data)//batch_size), (len(train_data)//batch_size)) - 1
    
    
    if training_style == "standard":
        for ind in range(len_t):
            # measure data loading time
            inputs = train_data[ind*batch_size:(ind+1)*batch_size]
            targets = labels[ind*batch_size:(ind+1)*batch_size]

            inputs_attack = attack_data[ind*batch_size:(ind+1)*batch_size]
            targets_attack = attack_label[ind*batch_size:(ind+1)*batch_size]

            data_time.update(time.time() - end)

            inputs, targets = inputs.to(device), targets.to(device)
            inputs_attack , targets_attack = inputs_attack.to(device), targets_attack.to(device)

            # compute output
            outputs = model(inputs)
            outputs_non = model(inputs_attack)

            classifier_input = torch.cat((inputs,inputs_attack))

#             comb_inputs_h = torch.cat((h_layer,h_layer_non))
            comb_inputs = torch.cat((outputs,outputs_non))

            if device == "cuda":
                comb_targets= torch.cat((targets,targets_attack)).view([-1,1]).type(torch.cuda.FloatTensor)
            else:
                comb_targets= torch.cat((targets,targets_attack)).view([-1,1]).type(torch.FloatTensor)

            attack_input = comb_inputs #torch.cat((comb_inputs,comb_targets),1)

            if device == "cuda":
                one_hot_tr = torch.from_numpy((np.zeros((attack_input.size()[0],outputs.size(1))))).to(device).type(torch.cuda.FloatTensor)
                target_one_hot_tr = one_hot_tr.scatter_(1, torch.cat((targets,targets_attack)).type(torch.cuda.LongTensor).view([-1,1]).data,1)
            else:
                one_hot_tr = torch.from_numpy((np.zeros((attack_input.size()[0],outputs.size(1))))).to(device).type(torch.FloatTensor)
                target_one_hot_tr = one_hot_tr.scatter_(1, torch.cat((targets,targets_attack)).type(torch.LongTensor).view([-1,1]).data,1)
            
            infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)

    #         sf= nn.Softmax(dim=0)
    #         att_inp=torch.stack([attack_input, infer_input_one_hot],1)
    #         att_inp = att_inp.view([attack_input.size()[0],1,2,attack_input.size(1)])

            #attack_output = attack_model(att_inp).view([-1])
            attack_output = attack_model(attack_input, infer_input_one_hot).view([-1])
            #attack_output = attack_model(attack_input).view([-1])
            att_labels = np.zeros((inputs.size()[0]+inputs_attack.size()[0]))
            att_labels [:inputs.size()[0]] =1.0
            att_labels [inputs.size()[0]:] =0.0
            is_member_labels = torch.from_numpy(att_labels).type(torch.FloatTensor)

            is_member_labels = is_member_labels.to(device)

            v_is_member_labels = torch.autograd.Variable(is_member_labels)

            if device == "cuda":
                classifier_targets = comb_targets.clone().view([-1]).type(torch.cuda.LongTensor)
            else:
                classifier_targets = comb_targets.clone().view([-1]).type(torch.LongTensor)        

            loss_attack = attack_criterion(attack_output, v_is_member_labels)

            # measure accuracy and record loss
            #prec1,p5 = accuracy(attack_output.data, v_is_member_labels.data, topk=(1,2))

            prec1=np.mean(np.equal((attack_output.data.cpu().numpy() >0.5),(v_is_member_labels.data.cpu().numpy()> 0.5)))
            losses.update(loss_attack.data, attack_input.size()[0])
            top1.update(prec1, attack_input.size()[0])

            # compute gradient and do SGD step
            attack_optimizer.zero_grad()
            loss_attack.backward()
            attack_optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            if verbose and ind % 100 == 0:
                print('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
                        batch=ind + 1,
                        size=len_t,
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        ))
    
    elif training_style == "coin_flip":
        inputs = train_data[i*batch_size:(i+1)*batch_size]
        targets = labels[i*batch_size:(i+1)*batch_size]

        inputs_attack = attack_data[i*batch_size:(i+1)*batch_size]
        targets_attack = attack_label[i*batch_size:(i+1)*batch_size]

        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)
        inputs_attack , targets_attack = inputs_attack.to(device), targets_attack.to(device)

        # compute output
        outputs = model(inputs)
        outputs_non = model(inputs_attack)

        classifier_input = torch.cat((inputs,inputs_attack))

#         comb_inputs_h = torch.cat((h_layer,h_layer_non))
        comb_inputs = torch.cat((outputs,outputs_non))

        if device == "cuda":
            comb_targets = torch.cat((targets,targets_attack)).view([-1,1]).type(torch.cuda.FloatTensor)
        else:
            comb_targets = torch.cat((targets,targets_attack)).view([-1,1]).type(torch.FloatTensor)

        attack_input = comb_inputs #torch.cat((comb_inputs,comb_targets),1)

        if device == "cuda":
            one_hot_tr = torch.from_numpy((np.zeros((attack_input.size()[0],outputs.size(1))))).to(device).type(torch.cuda.FloatTensor)
            target_one_hot_tr = one_hot_tr.scatter_(1, torch.cat((targets,targets_attack)).type(torch.cuda.LongTensor).view([-1,1]).data,1)
        else:
            one_hot_tr = torch.from_numpy((np.zeros((attack_input.size()[0],outputs.size(1))))).to(device).type(torch.FloatTensor)
            target_one_hot_tr = one_hot_tr.scatter_(1, torch.cat((targets,targets_attack)).type(torch.LongTensor).view([-1,1]).data,1)

        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)

#         sf= nn.Softmax(dim=0)
#         att_inp=torch.stack([attack_input, infer_input_one_hot],1)
#         att_inp = att_inp.view([attack_input.size()[0],1,2,attack_input.size(1)])

        #attack_output = attack_model(att_inp).view([-1])
        attack_output = attack_model(attack_input, infer_input_one_hot).view([-1])
        #attack_output = attack_model(attack_input).view([-1])
        att_labels = np.zeros((inputs.size()[0]+inputs_attack.size()[0]))
        att_labels[:inputs.size()[0]] = 1.0
        att_labels[inputs.size()[0]:] = 0.0
        is_member_labels = torch.from_numpy(att_labels).type(torch.FloatTensor)

        is_member_labels = is_member_labels.to(device)

        v_is_member_labels = torch.autograd.Variable(is_member_labels)

        if device == "cuda":
            classifier_targets = comb_targets.clone().view([-1]).type(torch.cuda.LongTensor)
        else:
            classifier_targets = comb_targets.clone().view([-1]).type(torch.LongTensor)

        loss_attack = attack_criterion(attack_output, v_is_member_labels)

        # measure accuracy and record loss
        #prec1,p5 = accuracy(attack_output.data, v_is_member_labels.data, topk=(1,2))

        prec1 = np.mean(np.equal((attack_output.data.cpu().numpy() >0.5),(v_is_member_labels.data.cpu().numpy()> 0.5)))
        losses.update(loss_attack.data, attack_input.size()[0])
        top1.update(prec1, attack_input.size()[0])
        
        # compute gradient and do SGD step
        attack_optimizer.zero_grad()
        loss_attack.backward()
        attack_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if verbose and i % 100 == 0:
            print  ('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
                    batch=i + 1,
                    size=len_t,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    ))
            
    elif training_style == "code":
        for ind in range(skip_batch, len_t):

            if ind >= skip_batch+num_batchs:
                break
            # measure data loading time
            inputs = train_data[ind*batch_size:(ind+1)*batch_size]
            targets = labels[ind*batch_size:(ind+1)*batch_size]

            inputs_attack = attack_data[ind*batch_size:(ind+1)*batch_size]
            targets_attack = attack_label[ind*batch_size:(ind+1)*batch_size]

            data_time.update(time.time() - end)

            inputs, targets = inputs.to(device), targets.to(device)
            inputs_attack , targets_attack = inputs_attack.to(device), targets_attack.to(device)

            # compute output
            outputs = model(inputs)
            outputs_non = model(inputs_attack)

            classifier_input = torch.cat((inputs,inputs_attack))

#             comb_inputs_h = torch.cat((h_layer,h_layer_non))
            comb_inputs = torch.cat((outputs,outputs_non))

            if device == "cuda":
                comb_targets = torch.cat((targets,targets_attack)).view([-1,1]).type(torch.cuda.FloatTensor)
            else:
                comb_targets = torch.cat((targets,targets_attack)).view([-1,1]).type(torch.FloatTensor)

            attack_input = comb_inputs #torch.cat((comb_inputs,comb_targets),1)

            if device == "cuda":
                one_hot_tr = torch.from_numpy((np.zeros((attack_input.size()[0],outputs.size(1))))).to(device).type(torch.cuda.FloatTensor)
                target_one_hot_tr = one_hot_tr.scatter_(1, torch.cat((targets,targets_attack)).type(torch.cuda.LongTensor).view([-1,1]).data,1)
            else:
                one_hot_tr = torch.from_numpy((np.zeros((attack_input.size()[0],outputs.size(1))))).to(device).type(torch.FloatTensor)
                target_one_hot_tr = one_hot_tr.scatter_(1, torch.cat((targets,targets_attack)).type(torch.LongTensor).view([-1,1]).data,1)

            infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)

    #         sf= nn.Softmax(dim=0)
    #         att_inp=torch.stack([attack_input, infer_input_one_hot],1)
    #         att_inp = att_inp.view([attack_input.size()[0],1,2,attack_input.size(1)])

            #attack_output = attack_model(att_inp).view([-1])
            attack_output = attack_model(attack_input, infer_input_one_hot).view([-1])
            #attack_output = attack_model(attack_input).view([-1])
            att_labels = np.zeros((inputs.size()[0]+inputs_attack.size()[0]))
            att_labels [:inputs.size()[0]] =1.0
            att_labels [inputs.size()[0]:] =0.0
            is_member_labels = torch.from_numpy(att_labels).type(torch.FloatTensor)

            is_member_labels = is_member_labels.to(device)

            v_is_member_labels = torch.autograd.Variable(is_member_labels)

            if device == "cuda":
                classifier_targets = comb_targets.clone().view([-1]).type(torch.cuda.LongTensor)
            else:
                classifier_targets = comb_targets.clone().view([-1]).type(torch.LongTensor)

            loss_attack = attack_criterion(attack_output, v_is_member_labels)

            # measure accuracy and record loss
            #prec1,p5 = accuracy(attack_output.data, v_is_member_labels.data, topk=(1,2))

            prec1=np.mean(np.equal((attack_output.data.cpu().numpy() >0.5),(v_is_member_labels.data.cpu().numpy()> 0.5)))
            losses.update(loss_attack.data, attack_input.size()[0])
            top1.update(prec1, attack_input.size()[0])

            # compute gradient and do SGD step
            attack_optimizer.zero_grad()
            loss_attack.backward()
            attack_optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            if verbose and ind % 100 == 0:
                print('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
                        batch=ind + 1,
                        size=len_t,
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        ))
    else:
        raise ValueError("unhandled training style")

    return (losses.avg, top1.avg)


def test_attack(train_data, labels, attack_data, attack_label, model, attack_model, criterion, attack_criterion, optimizer, 
                attack_optimizer, batch_size, epoch, device, verbose=False):

    model.eval()
    attack_model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    end = time.time()
    len_t =  min((len(attack_data)//batch_size) ,(len(train_data)//batch_size))-1
    member_prob = np.zeros((len_t+1)*batch_size)
    nonmember_prob = np.zeros((len_t+1)*batch_size)
    for ind in range(len_t):
        # measure data loading time
        inputs = train_data[ind*batch_size:(ind+1)*batch_size]
        targets = labels[ind*batch_size:(ind+1)*batch_size]
        
        inputs_attack = attack_data[ind*batch_size:(ind+1)*batch_size]
        targets_attack = attack_label[ind*batch_size:(ind+1)*batch_size]
        
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)
        inputs_attack , targets_attack = inputs_attack.to(device), targets_attack.to(device)

        # compute output
        outputs = model(inputs)
        outputs_non = model(inputs_attack)
        
#         comb_inputs_h = torch.cat((h_layer,h_layer_non))
        comb_inputs = torch.cat((outputs,outputs_non))
        
        if device == "cuda":
            comb_targets = torch.cat((targets,targets_attack)).view([-1,1]).type(torch.cuda.FloatTensor)
        else:
            comb_targets = torch.cat((targets,targets_attack)).view([-1,1]).type(torch.FloatTensor)
            
        attack_input = comb_inputs #torch.cat((comb_inputs,comb_targets),1)

        
        if device == "cuda":
            one_hot_tr = torch.from_numpy((np.zeros((attack_input.size()[0],outputs.size(1))))).to(device).type(torch.cuda.FloatTensor)
            target_one_hot_tr = one_hot_tr.scatter_(1, torch.cat((targets,targets_attack)).type(torch.cuda.LongTensor).view([-1,1]).data,1)
        else:
            one_hot_tr = torch.from_numpy((np.zeros((attack_input.size()[0],outputs.size(1))))).to(device).type(torch.FloatTensor)
            target_one_hot_tr = one_hot_tr.scatter_(1, torch.cat((targets,targets_attack)).type(torch.LongTensor).view([-1,1]).data,1)
        
        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)
        
        #attack_output = attack_model(att_inp).view([-1])
        attack_output = attack_model(attack_input, infer_input_one_hot).view([-1])
        #attack_output = attack_model(attack_input).view([-1])
        att_labels = np.zeros((inputs.size()[0]+inputs_attack.size()[0]))
        att_labels [:inputs.size()[0]] =1.0
        att_labels [inputs.size()[0]:] =0.0
        is_member_labels = torch.from_numpy(att_labels).type(torch.FloatTensor)
        is_member_labels = is_member_labels.to(device)
        
        v_is_member_labels = torch.autograd.Variable(is_member_labels)
        
        loss = attack_criterion(attack_output, v_is_member_labels)
        
        # measure accuracy and record loss
        #prec1,p5 = accuracy(attack_output.data, v_is_member_labels.data, topk=(1,2))
        member_prob[ind*batch_size:(ind+1)*batch_size] = attack_output.data.cpu().numpy()[:batch_size]
        nonmember_prob[ind*batch_size:(ind+1)*batch_size] = attack_output.data.cpu().numpy()[batch_size:]
        prec1 = np.mean(np.equal((attack_output.data.cpu().numpy() > 0.5),(v_is_member_labels.data.cpu().numpy() > 0.5)))
        losses.update(loss.data, attack_input.size()[0])
        top1.update(prec1, attack_input.size()[0])
        
        #raise
        
        # compute gradient and do SGD step

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if verbose and ind % 100 == 0:
            print  ('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
                    batch=ind + 1,
                    size=len_t,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    ))
            

    return (losses.avg, top1.avg, np.mean(member_prob), np.mean(nonmember_prob))
        
        