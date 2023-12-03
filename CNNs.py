# -*- coding: utf-8 -*-
"""
@file: CNNs.py
@description: 
"""
import torch
import torch.nn as nn
import CNNModels as models
from CNNModels import ResNet50_Weights, ResNet101_Weights, ResNet152_Weights, ResNet34_Weights, ResNet18_Weights
from parameters import * 

def TypeIILinear(x):
    #be released after final decision
    return x

def Gaussian(x):
    #be released after decision
    return x

def Impulse(x,prob): #salt_and_pepper noise
    #be released after decision
    return x


class ResNet152(nn.Module):
    def __init__(self, num_classes, checkpoint, noise_type, noise_strength, noisy_layer, sub_noisy_layer, Pretrain = False):
        super(ResNet152, self).__init__()
        self.noisy_layer = noisy_layer
        self.noise_str = noise_strength
        self.sub_noisy_layer = sub_noisy_layer
        self.noise_type = noise_type
        self.model = models.resnet152(weights=None)  
        if(Pretrain):
            if(checkpoint):
                self.model.load_state_dict(torch.load(checkpoint))
            self.model = models.resnet152(weights=ResNet152_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
 
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        if(self.noisy_layer == 1):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer1)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        x = TypeIILinear(x)
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer1[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer1[temp_sub](x)
                else:
                    x = self.model.layer1[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer1(x)
        if(self.noisy_layer == 2):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer2)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        x = TypeIILinear(x)
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer2[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer2[temp_sub](x)
                else:
                    x = self.model.layer2[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer2(x)
        if(self.noisy_layer == 3):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer3)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        x = TypeIILinear(x)
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer3[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer3[temp_sub](x)
                else:
                    x = self.model.layer3[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer3(x)
        if(self.noisy_layer == 4):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer4)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        x = TypeIILinear(x)
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer4[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer4[temp_sub](x)
                else:
                    x = self.model.layer4[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        # x = x.view(x.size(0), x.size(1))
        return x

class ResNet101(nn.Module):
    def __init__(self, num_classes, checkpoint, noise_type, noise_strength, noisy_layer, sub_noisy_layer, Pretrain = False):
        super(ResNet101, self).__init__()
        self.noisy_layer = noisy_layer
        self.noise_str = noise_strength
        self.sub_noisy_layer = sub_noisy_layer
        self.noise_type = noise_type
        self.model = models.resnet101(weights=None)  
        if(Pretrain):
            if(checkpoint):
                self.model.load_state_dict(torch.load(checkpoint))
            self.model = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
 
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        if(self.noisy_layer == 1):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer1)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        x = TypeIILinear(x)
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer1[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer1[temp_sub](x)
                else:
                    x = self.model.layer1[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer1(x)
        if(self.noisy_layer == 2):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer2)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        x = TypeIILinear(x)
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer2[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer2[temp_sub](x)
                else:
                    x = self.model.layer2[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer2(x)
        if(self.noisy_layer == 3):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer3)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        x = TypeIILinear(x)
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer3[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer3[temp_sub](x)
                else:
                    x = self.model.layer3[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer3(x)
        if(self.noisy_layer == 4):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer4)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        x = TypeIILinear(x)
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer4[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer4[temp_sub](x)
                else:
                    x = self.model.layer4[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        # x = x.view(x.size(0), x.size(1))
        return x
 
class ResNet50(nn.Module):
    def __init__(self, num_classes, checkpoint, noise_type, noise_strength, noisy_layer, sub_noisy_layer, Pretrain = False):
        super(ResNet50, self).__init__()
        self.noisy_layer = noisy_layer
        self.noise_str = noise_strength
        self.sub_noisy_layer = sub_noisy_layer
        self.noise_type = noise_type
        self.model = models.resnet50(weights=None)  
        if(Pretrain):
            if(checkpoint):
                self.model.load_state_dict(torch.load(checkpoint))
            self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
 
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        if(self.noisy_layer == 1):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer1)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        x = TypeIILinear(x)
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer1[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer1[temp_sub](x)
                else:
                    x = self.model.layer1[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer1(x)
        if(self.noisy_layer == 2):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer2)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        x = TypeIILinear(x)
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer2[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer2[temp_sub](x)
                else:
                    x = self.model.layer2[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer2(x)
        if(self.noisy_layer == 3):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer3)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        x = TypeIILinear(x)
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer3[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer3[temp_sub](x)
                else:
                    x = self.model.layer3[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer3(x)
        if(self.noisy_layer == 4):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer4)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        x = TypeIILinear(x)
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer4[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer4[temp_sub](x)
                else:
                    x = self.model.layer4[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        # x = x.view(x.size(0), x.size(1))
        return x
    
class ResNet34(nn.Module):
    def __init__(self, num_classes, checkpoint, noise_type, noise_strength, noisy_layer, sub_noisy_layer, Pretrain = False):
        super(ResNet34, self).__init__()
        self.noisy_layer = noisy_layer
        self.noise_str = noise_strength
        self.sub_noisy_layer = sub_noisy_layer
        self.noise_type = noise_type
        self.model = models.resnet34(weights=None)  
        if(Pretrain):
            if(checkpoint):
                self.model.load_state_dict(torch.load(checkpoint))
            self.model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
 
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        if(self.noisy_layer == 1):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer1)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        x = TypeIILinear(x)
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer1[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer1[temp_sub](x)
                else:
                    x = self.model.layer1[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer1(x)
        if(self.noisy_layer == 2):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer2)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        x = TypeIILinear(x)
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer2[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer2[temp_sub](x)
                else:
                    x = self.model.layer2[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer2(x)
        if(self.noisy_layer == 3):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer3)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        x = TypeIILinear(x)
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer3[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer3[temp_sub](x)
                else:
                    x = self.model.layer3[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer3(x)
        if(self.noisy_layer == 4):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer4)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear2'):
                        x = TypeIILinear(x)
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer4[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer4[temp_sub](x)
                else:
                    x = self.model.layer4[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        # x = x.view(x.size(0), x.size(1))
        return x
    
class ResNet18(nn.Module):
    def __init__(self, num_classes, checkpoint, noise_type, noise_strength, noisy_layer, sub_noisy_layer, Pretrain = False):
        super(ResNet18, self).__init__()
        self.noisy_layer = noisy_layer
        self.noise_str = noise_strength
        self.sub_noisy_layer = sub_noisy_layer
        self.noise_type = noise_type
        self.model = models.resnet18(weights=None)  
        if(Pretrain):
            if(checkpoint):
                self.model.load_state_dict(torch.load(checkpoint))
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
 
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        if(self.noisy_layer == 1):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer1)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        x = TypeIILinear(x)
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer1[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer1[temp_sub](x)
                else:
                    x = self.model.layer1[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer1(x)
        if(self.noisy_layer == 2):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer2)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        x = TypeIILinear(x)
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer2[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer2[temp_sub](x)
                else:
                    x = self.model.layer2[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer2(x)
        if(self.noisy_layer == 3):
            if(self.sub_noisy_layer == 1):
                x_copy = x.detach()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer3)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        x = TypeIILinear(x)
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer3[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer3[temp_sub](x)
                else:
                    x = self.model.layer3[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer3(x)
        if(self.noisy_layer == 4):
            if(self.sub_noisy_layer == 1):
                x_copy = x.clone()
                x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
            for temp_sub in range(len(self.model.layer4)):
                if( self.sub_noisy_layer == temp_sub+1):
                    if(self.noise_type == 'linear'):
                        x = TypeIILinear(x)
                    elif(self.noise_type == 'gaussian'):
                        x = Gaussian(x)
                        x = self.model.layer4[temp_sub](x)
                    elif(self.noise_type == 'impulse'):
                        x = Impulse(x, self.noise_str)
                        x = self.model.layer4[temp_sub](x)
                else:
                    x = self.model.layer4[temp_sub](x)
                    x_copy = x.detach()
                    x_copy = torch.cat( (x_copy[1:,:,:,:], x_copy[0,:,:,:].unsqueeze(0)),dim=0 )
        else:
            x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        # x = x.view(x.size(0), x.size(1))
        return x
