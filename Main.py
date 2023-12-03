
import CNNs 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import os,random
import numpy as np
from torch.utils.data import DataLoader
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import tqdm
from dataloader import *
from parameters import *

seed_everything(42)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if cuda else 'cpu')

#
if args.resnet == 'resnet18':
    noisy_cnn = CNNs.ResNet18(num_classes=args.class_num, Pretrain=args.pretrain, checkpoint = None, noise_type = args.noise_type, noise_strength = args.noise_str, noisy_layer = args.noise_layer, sub_noisy_layer = args.sub_noisy_layer)
elif args.resnet == 'resnet34':
    noisy_cnn = CNNs.ResNet34(num_classes=args.class_num, Pretrain=args.pretrain, checkpoint = None, noise_type = args.noise_type, noise_strength = args.noise_str, noisy_layer = args.noise_layer, sub_noisy_layer = args.sub_noisy_layer)
elif args.resnet == 'resnet50':
    noisy_cnn = CNNs.ResNet50(num_classes=args.class_num, Pretrain=args.pretrain, checkpoint = None, noise_type = args.noise_type, noise_strength = args.noise_str, noisy_layer = args.noise_layer, sub_noisy_layer = args.sub_noisy_layer)
elif args.resnet == 'resnet101':
    noisy_cnn = CNNs.ResNet101(num_classes=args.class_num, Pretrain=args.pretrain, checkpoint = None, noise_type = args.noise_type, noise_strength = args.noise_str, noisy_layer = args.noise_layer, sub_noisy_layer = args.sub_noisy_layer)
elif args.resnet == 'resnet152':
    noisy_cnn = CNNs.ResNet152(num_classes=args.class_num, Pretrain=True, checkpoint = None, noise_type = args.noise_type, noise_strength = args.noise_str, noisy_layer = args.noise_layer, sub_noisy_layer = args.sub_noisy_layer)
#if pretrain = True, user can desigate checkpoint, or it will use default weights
print('model parameters:',sum(param.numel() for param in noisy_cnn.parameters())/1e6)

'''
model_dict = model.state_dict()
pretrained = torch.load('./resnet50-19c8e357.pth')
print(len(list(pretrained.keys())))
print(len(list(model_dict.keys())))
pretrained_dict = {k: v for k, v in pretrained.items() if k in model_dict}
print(pretrained_dict)
'''

#
#
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_test = transforms.Compose([    
    transforms.RandomResizedCrop(224), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#

if(args.datasets == 'cifar10'):
    train_data = datasets.CIFAR10(root='./data', train=True,transform=transform_train,download=True)
    test_data =datasets.CIFAR10(root='./data',train=False,transform=transform_test,download=True)
elif(args.datasets == 'cifar100'):
    train_data = datasets.CIFAR100(root='./data', train=True,transform=transform_train,download=True)
    test_data =datasets.CIFAR100(root='./data',train=False,transform=transform_test,download=True)
elif(args.datasets == 'TinyImageNet'):
    train_loader, test_loader = get_loader(args.tinyImagenet_path)
elif(args.datasets == 'ImageNet'):
    train_loader, test_loader = get_loader(args.Imagenet_path)

if(args.datasets == 'cifar10' or args.datasets == 'cifar100'):
    train_loader = DataLoader(dataset=train_data,batch_size=args.batch_size,shuffle=True,num_workers=0)
    test_loader = DataLoader(dataset=test_data,batch_size=args.batch_size,shuffle=False,num_workers=0)


criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(noisy_cnn.parameters(),lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, eta_min=1e-9, last_epoch=- 1, verbose=False)

noisy_cnn.to(device)
best_acc = 0

for epoch in range(args.epoch):
    noisy_cnn.train()
    total = 0
    correct = 0
    for i,data in enumerate(tqdm.tqdm(train_loader)):
       
        inputs,labels = data
        
        inputs,labels = inputs.to(device),labels.to(device)
        
       
        outputs = noisy_cnn(inputs)
     
        loss = criterion(outputs,labels)
        _, predictions = torch.max(outputs, 1)
        total += labels.size(0)
        correct = correct +(predictions == labels).sum().item()
      
        optimizer.zero_grad()
        
  
        loss.backward()
  
        optimizer.step()
        #
        #if  i%50 == 10:
        #    print("Train/BatchIters/Acc", loss.item(), epoch*len(train_loader)+i, correct/total)
        #print('it’s training...{}'.format(i))
    print('epoch {} loss:{:.4f} Acc:{:3f}'.format(epoch+1,loss.item(), correct/total))
    scheduler.step() 

    #
    noisy_cnn.eval()
    correct,total = 0,0
    for j,data in enumerate(tqdm.tqdm(test_loader)):
        inputs,labels = data
        inputs,labels = inputs.to(device),labels.to(device)
     
        with torch.no_grad():
            outputs = noisy_cnn(inputs)
        _, predicted = torch.max(outputs.data,1)
        total =total+labels.size(0)
        correct = correct +(predicted == labels).sum().item()
        #
        #if  j%20 == 10:
        #    print("Accuracy/Test Iters", 100.0*correct/total, j)
            
    print('Accuracy: {:.4f}%'.format(100.0*correct/total))

   
    if(correct/total > best_acc):
        if(not os.path.exists('./saved_models')):
            os.mkdir('./saved_models')
        best_acc = correct/total
        if(args.noise_str == 0):
            torch.save(noisy_cnn.state_dict(), './saved_models/'+str(correct/total)+'_'+args.datasets+'_'+args.resnet+'_pretrain_'+str(args.pretrain)+'_vanilla'+'.pth')
        else:
            if(args.noise_type != 'gaussian'):
                torch.save(noisy_cnn.state_dict(), './saved_models/'+str(correct/total)+'_'+args.datasets+'_'+args.resnet+'_pretrain_'+str(args.pretrain)+'_noiseType_'+str(args.noise_type)+'_str_'+str(args.noise_str)+'_layer_'+str(args.noise_layer)+'_sublayer_'+str(args.sub_noisy_layer)+'.pth')
            elif(args.noise_type == 'gaussian'):
                torch.save(noisy_cnn.state_dict(), './saved_models/'+str(correct/total)+'_'+args.datasets+'_'+args.resnet+'_pretrain_'+str(args.pretrain)+'_noiseType_'+str(args.noise_type)+'_str_'+str(args.noise_str)+'_layer_'+str(args.noise_layer)+'_sublayer_'+str(args.sub_noisy_layer)+\
                    '_'+str(args.gau_mean)+'_'+str(args.gau_var)+'.pth')
        print('model saved')
        files = os.listdir('./saved_models')
        for subfile in files:
            noisy = (subfile.split('_')[5]).split('.')[0]
            dataset = (subfile.split('_')[1])
            model = (subfile.split('_')[2])
            if (dataset != args.datasets or model != args.resnet):
                continue
            if(noisy != 'vanilla'):
                sub_layer = (subfile.split('_')[12]).split('.')[0]
                layer = subfile.split('_')[10]
                if(args.datasets in subfile and args.resnet in subfile and args.noise_type in subfile and str(args.sub_noisy_layer) == sub_layer and str(args.noise_layer) == layer and str(args.noise_str) in subfile):
                    saved_acc = float(subfile.split('_')[0])
                    if(args.noise_type == 'gaussian'):
                        gau_mean = float(subfile.split('_')[13])
                        gau_var = float(subfile.split('_')[14][0:3])
                        if (saved_acc<best_acc and (args.gau_mean) == gau_mean and (args.gau_var) == gau_var ):
                            os.remove('./saved_models/'+subfile)
                    else:
                        if (saved_acc<best_acc ):
                            os.remove('./saved_models/'+subfile)
            elif(noisy == 'vanilla'):
                sub_layer = 'NA'
                layer = 'NA'
                if(args.datasets in subfile and args.resnet in subfile and args.noise_type not in subfile and noisy in subfile):
                    saved_acc = float(subfile.split('_')[0])
                    if (saved_acc<best_acc ):
                        os.remove('./saved_models/'+subfile)
            
 
  
print('Finish training!')
