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
from torch.autograd import Variable
from torchsummary import summary
from Randomizer import SRR
import time

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Convolutional Layers
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc_layers = nn.Sequential(
            # Fully Connected Layers
            nn.Linear(64 * 8 * 8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x



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
            avg_params[param_name] = torch.mean(torch.stack([params[param_name].float() for params in model_params]), 0)
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
            avg_params[param_name] = torch.mean(torch.stack([params[param_name].float() for params in model_params]), 0)
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
            avg_params[param_name] = torch.mean(torch.stack([params[param_name].float() for params in model_params]), 0)
        global_model.load_state_dict(avg_params)
        
        for model in all_models:
            model.load_state_dict(global_model.state_dict())
        
        return
    
    def server_aggregate_srr(self, global_model, client_models, all_models, ldp_func):
        
        
        with torch.no_grad():
            for model in client_models:
                for param in model.parameters():
                    param_val = param.data.detach().cpu().flatten()
                    perturbed_weight = [ldp_func.perturb(val) for val in param_val]
                    param.data = torch.tensor(perturbed_weight).reshape(param.shape).to(param.device)
        
        model_params = [model.state_dict() for model in client_models]
        avg_params = {}
        for param_name in model_params[0]:
            avg_params[param_name] = torch.mean(torch.stack([params[param_name].float() for params in model_params]), 0)
        global_model.load_state_dict(avg_params)
        
        for model in all_models:
            model.load_state_dict(global_model.state_dict())
        
        return

    
    def server_aggregate_srr_adaptive(self, global_model, client_models, all_models, ldp_funcs):
        
        
        with torch.no_grad():
            for i, model in enumerate(client_models):
                with open('cifar-10-experiment_SRR.txt', 'a+') as f:
                    start = time.time()
                    for name, param in model.named_parameters():
                        t1 = time.time()
                        param_val = param.data.detach().cpu().flatten().numpy()
                        t2 = time.time()
                        for idx in range(len(param_val)):
                            param_val[idx] = ldp_funcs[name].perturb(param_val[idx])
                        t3 = time.time()
                        param.data = torch.tensor(param_val).reshape(param.shape).to(param.device)
                        t4 = time.time()
                    end = time.time()
                    print(f"Model {i+1} perturbed!")
                    print(f"Time taken for perturbation : {end-start} seconds.\n")
                    f.write(f"Time taken for perturbation of Model {i+1} : {end-start} seconds.\n")
        
        model_params = [model.state_dict() for model in client_models]
        avg_params = {}
        for param_name in model_params[0]:
            avg_params[param_name] = torch.mean(torch.stack([params[param_name].float() for params in model_params]), 0)
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
