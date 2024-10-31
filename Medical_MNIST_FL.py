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
from torch.autograd import Variable
from torchsummary import summary
from Randomizer import SRR
from LDP_Functions import LDP_FL

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d(3,6,3,1)
        self.conv2=nn.Conv2d(6,16,3,1)
        self.fc1=nn.Linear(16*54*54,120) 
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,20)
        self.fc4=nn.Linear(20,6)
    
    def forward(self,X):
        X=F.relu(self.conv1(X))
        X=F.max_pool2d(X,2,2)
        X=F.relu(self.conv2(X))
        X=F.max_pool2d(X,2,2)
        X=X.view(-1,16*54*54)
        X=F.relu(self.fc1(X))
        X=F.relu(self.fc2(X))
        X=F.relu(self.fc3(X))
        X=self.fc4(X)
        
        return F.log_softmax(X,dim=1)


class FedMLFunc:

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
                    param_val = param.data.detach().cpu().flatten().numpy()
                    for idx in range(len(param_val)):
                        param_val[idx] = ldp_func.perturb(param_val[idx])
                    param.data = torch.tensor(param_val).reshape(param.shape).to(param.device)
        
        model_params = [model.state_dict() for model in client_models]
        avg_params = {}
        for param_name in model_params[0]:
            avg_params[param_name] = torch.mean(torch.stack([params[param_name] for params in model_params]), 0)
        global_model.load_state_dict(avg_params)
        
        for model in all_models:
            model.load_state_dict(global_model.state_dict())
        
        return

    
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
