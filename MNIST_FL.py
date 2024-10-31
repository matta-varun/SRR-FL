#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This code was utilized and modified from the version available at
# https://github.com/yonetaniryo/federated_learning_pytorch/blob/master/FL_pytorch.ipynb
# with the following license.

# MIT License

# Copyright (c) 2021 Ryo Yonetani

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn.functional as F
import torch.nn as nn
import math as math
import time
import random
import numpy as np
from torch.autograd import Variable
from torchsummary import summary
from Randomizer import SRR
from LDP_Functions import LDP_FL
import statistics as s

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc1(x)
        return x



class FedMLFunc:
    
    def __init__(self):
        self.history = {}
    
    def initialize_history(self, model):
        print(f"Numpy test : {np.mean([1,2,3])}")
        with torch.no_grad():
            for name, param in model.named_parameters():
                param_val = param.data.detach().cpu().flatten().numpy()
                self.history[name] = []
                for idx in range(len(param_val)):
                    self.history[name].append([])

    def client_update(self, net, optimizer, trainloader, epochs, DEVICE):
        
        criterion = torch.nn.CrossEntropyLoss()
        net.train()
        tot_total = 0
        tot_correct = 0
        for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for images, labels in trainloader:
                images, labels = images.cuda(), labels.cuda()
                outputs = net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Metrics
                epoch_loss += loss.item()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            tot_total += total
            tot_correct += correct
            epoch_loss /= len(trainloader.dataset)
            epoch_acc = correct / total
            torch.cuda.empty_cache()
        
        return [loss.item(),tot_correct/tot_total]

    
    def server_aggregate(self, global_model, client_models, all_models):
        
        model_params = [model.state_dict() for model in client_models]
        avg_params = {}
        for param_name in model_params[0]:
            avg_params[param_name] = torch.mean(torch.stack([params[param_name] for params in model_params]), 0)
        global_model.load_state_dict(avg_params)
        
        for model in all_models:
            model.load_state_dict(global_model.state_dict())
        
        return
    
    def server_aggregate_attack(self, global_model, global_copy, client_models, all_models, ldp_func, error_bound = 0.01, score = 70.0):
        
        print("SERVER AGGREGATION BEGINS\n")
        
        model_params = [model.state_dict() for model in client_models]
        avg_params = {}
        for param_name in model_params[0]:
            avg_params[param_name] = torch.mean(torch.stack([params[param_name] for params in model_params]), 0)
        global_copy.load_state_dict(avg_params)
        
        print("Global Copy updated.")
        
        count_dict = {}
        
        with torch.no_grad():
            for i, model in enumerate(client_models):
                for name, param in model.named_parameters():
                    if name not in count_dict:
                        count_dict[name] = []
                    param_val = param.data.detach().cpu().flatten().numpy()
                    if len(count_dict[name]) != len(param_val):
                        for idx in range(len(param_val)):
                            count_dict[name].append([0, 0])
                    for idx in range(len(param_val)):
                        param_val[idx] = ldp_func.perturb(param_val[idx])
                        count_dict[name][idx][ldp_func.check(param_val[idx])] += 1
                    param.data = torch.tensor(param_val).reshape(param.shape).to(param.device)
                print(f'Model {i+1} perturbed!')
        
        # History updated
        for name, param in global_model.named_parameters():
            param_val = param.data.detach().cpu().flatten().numpy()
            for idx in range(len(param_val)):
                self.history[name][idx].append(count_dict[name][idx])
        
        model_params = [model.state_dict() for model in client_models]
        avg_params = {}
        for param_name in model_params[0]:
            avg_params[param_name] = torch.mean(torch.stack([params[param_name] for params in model_params]), 0)
        global_model.load_state_dict(avg_params)
        
        print("Global Model updated.")
        
        with torch.no_grad():
            success_count = 0
            mean_err = 0
            scores = 0
            mean_temp = 0
            for name, param in global_copy.named_parameters():
                param_val = param.data.detach().cpu().flatten().numpy()
                for idx in range(len(param_val)):
                    # get history and compute current pred and confidence score
                    curr_hist = self.history[name][idx]
                    if len(curr_hist) < 2:
                        continue
                    else:
                        
                        vals2 = [v[0] for v in curr_hist]
                        vals1 = [v[1] for v in curr_hist]
                        meanval2 = np.mean(vals2)
                        meanval1 = np.mean(vals1)
                        hist_est_mean = ldp_func.estimate([meanval2, meanval1])
                        
                        random.shuffle(curr_hist)
                        split_point = len(curr_hist)//2

                        list1 = curr_hist[:split_point]
                        list2 = curr_hist[split_point:]

                        vals2 = [v[0] for v in list1]
                        vals1 = [v[1] for v in list1]
                        meanval2 = np.mean(vals2)
                        meanval1 = np.mean(vals1)
                        split_est_mean1 = ldp_func.estimate([meanval2, meanval1])

                        vals2 = [v[0] for v in list2]
                        vals1 = [v[1] for v in list2]
                        meanval2 = np.mean(vals2)
                        meanval1 = np.mean(vals1)
                        split_est_mean2 = ldp_func.estimate([meanval2, meanval1])
                        
                        temp = abs(split_est_mean1 - split_est_mean2)/(ldp_func.get_domain_size())
                        
                        confidence_score = round(1/(0.01 + temp), 4)
                        
                        if confidence_score > score:
                            curr_err = ( abs(param_val[idx] - hist_est_mean) )/(ldp_func.get_domain_size())
                            mean_err += curr_err
                            mean_temp += temp
                            success_count += 1
                            scores += confidence_score
            
            if success_count > 0:
                mean_err /= success_count
                scores /= success_count
                mean_temp /= success_count
                print(f"Mean Confidence Score above {score} was {scores}.")
                print(f"Mean Split Mean Difference was {mean_temp}.")
       
        for model in all_models:
            model.load_state_dict(global_model.state_dict())
        
        
        return mean_err, success_count                    
        
        
        # now we check with previous history and current global model weights for convergence
        with torch.no_grad():
            for name, param in global_model.named_parameters():
                param_val = param.data.detach().cpu().flatten().numpy()
                for idx in range(len(param_val)):
                    if param_val[idx] > self.history[name][idx][0]-error_bound and param_val[idx] < self.history[name][idx][0]+error_bound:
                        self.history[name][idx][1] = True
                    else:
                        self.history[name][idx][1] = False
                    self.history[name][idx][0] = param_val[idx]
#                     self.history[name][idx][2].append(param_val[idx])
                    if tuple(count_dict[name][idx]) not in self.history[name][idx][2]:
                        self.history[name][idx][2][tuple(count_dict[name][idx])] = 0
                    self.history[name][idx][2][ tuple(count_dict[name][idx]) ] += 1
        
        print("History and Status updated.")
        
        tot_count = 0
        method1 = 0
        method2 = 0
        pred_count = 0
        
        with torch.no_grad():
            for name, param in global_copy.named_parameters():
                param_val = param.data.detach().cpu().flatten().numpy()
                for idx in range(len(param_val)):
                    maxfreq = 0
                    value = [1,1]
                    for key, val in self.history[name][idx][2].items():
                        if val > maxfreq:
                            maxfreq = val
                            value = key
                    w_est = ldp_func.estimate(list(value))
                    if abs(param_val[idx]-w_est) <= 2*error_bound:
                        method1 += 1
                    tot_count += 1
                    
        print("Attack results computed.")
        
        for model in all_models:
            model.load_state_dict(global_model.state_dict())
        
        return tot_count, method1 #, method2, pred_count
    

    def server_aggregate_ldpfl(self, global_model, client_models, all_models):
        
        with torch.no_grad():
            for model in client_models:
                for param in model.parameters():
                    start = time.time()
                    param_value = param.data.flatten()
                    processed_param_value = torch.tensor([LDP_FL(val, c = 0, r = 0.075, epsilon = 1) for val in param_value])
                    new_param_value = processed_param_value.reshape(param.size())
                    param.data.copy_(new_param_value)
                    end =time.time()
                    print(f"Time taken for perturbing {len(param_value)} values using LDP-FL : {end-start} seconds.")
                print('Model perturbed!')
        
        model_params = [model.state_dict() for model in client_models]
        avg_params = {}
        for param_name in model_params[0]:
            avg_params[param_name] = torch.mean(torch.stack([params[param_name] for params in model_params]), 0)
        global_model.load_state_dict(avg_params)
        
        for model in all_models:
            model.load_state_dict(global_model.state_dict())
        
        return
    
    def server_aggregate_grr(self, global_model, client_models, all_models, ldp_func):
        
        with torch.no_grad():
            for model in client_models:
                for param in model.parameters():
                    param_val = param.data.detach().cpu().flatten()
                    perturbed_weight = [ldp_func.perturb(val) for val in param_val]
                    param.data = torch.tensor(perturbed_weight).reshape(param.shape).to(param.device)
        
        model_params = [model.state_dict() for model in client_models]
        avg_params = {}
        for param_name in model_params[0]:
            avg_params[param_name] = torch.mean(torch.stack([params[param_name] for params in model_params]), 0)
        global_model.load_state_dict(avg_params)
        
        for model in all_models:
            model.load_state_dict(global_model.state_dict())
        
        return
    
    def server_aggregate_srr(self, global_model, client_models, all_models, ldp_func):
        
        
        with torch.no_grad():
            for model in client_models:
                for param in model.parameters():
                    param_val = param.data.detach().cpu().flatten()
                    perturbed_weight = [ldp_func._perturb(val) for val in param_val]
                    param.data = torch.tensor(perturbed_weight).reshape(param.shape).to(param.device)

        
        model_params = [model.state_dict() for model in client_models]
        avg_params = {}
        for param_name in model_params[0]:
            avg_params[param_name] = torch.mean(torch.stack([params[param_name] for params in model_params]), 0)
        global_model.load_state_dict(avg_params)
        
        for model in all_models:
            model.load_state_dict(global_model.state_dict())
        
        return
        
        
        num_models = len(client_models)
        
        with torch.no_grad():
            for name, param in global_model.named_parameters():
                mean_param = sum(model.state_dict()[name] for model in client_models) / num_models

                param.copy_(mean_param)
        
        
        for i, model in enumerate(client_models):
            model.load_state_dict(global_model.state_dict())

    
    def test(self, net, testloader, DEVICE):
        
        net.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in testloader:
                data = data.cuda()
                target = target.cuda()
                output = net(data)
                loss = criterion(output, target)
                test_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0) 

        test_loss /= total
        test_accuracy = correct / total

        return test_loss, test_accuracy
