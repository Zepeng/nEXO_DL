#!/usr/bin/env python
# coding: utf-8

#Dataset code copied from https://github.com/utkuozbulak/pytorch-custom-dataset-examples
#model code copied from https://github.com/DeepLearnPhysics/pytorch-uresnet

import numpy as np
import yaml
from numpy import load
import os

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os

import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

import argparse
#from networks.preact_resnet import PreActResNet18
from networks.resnet_example import resnet14, resnet18
from utils.data_loaders import DenseDataset
from utils.data_loaders import DatasetFromSparseMatrix
import traceback
import networks.resnet as resnet

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
list_of_gpus = range(torch.cuda.device_count())
device_ids = range(torch.cuda.device_count())
os.environ["CUDA_VISIBLE_DEVICES"] = str(list_of_gpus).strip('[]').replace(' ', '')

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
epochs = 200

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = lr/np.exp(epoch/10)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def yaml_load(config):
    with open(config) as stream:
        param = yaml.safe_load(stream)
    return param

# Training
def train(trainloader, epoch):
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx%100 == 0:
            print(batch_idx, '/', len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/len(trainloader), 100.*correct/total

def test(testloader, epoch, saveall=False):

    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    score = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            softmax = nn.Softmax(dim=0)
            for m in range(outputs.size(0)):
                score.append([softmax(outputs[m])[1].item(), targets[m].item()])
                # score.append([outputs[m][1].item(), targets[m].item()])
            if batch_idx%100 == 0:
                print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    is_better = False
    acc = 100.*correct/total

    # If we want to save all training records
    if saveall:
        print ('Saving...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint_sens'):
            os.mkdir('checkpoint_sens' )
        if not os.path.isdir('training_outputs'):
            os.mkdir('training_outputs')
        torch.save(state, './checkpoint_sens/ckpt_%d.t7' % epoch)
        torch.save(state, './checkpoint_sens/ckpt.t7' )
        best_acc = acc
    # Otherwise only save the best one
    elif acc > best_acc:
        print ('Saving...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint_sens'):
            os.mkdir('checkpoint_sens' )
        if not os.path.isdir('training_outputs'):
            os.mkdir('training_outputs')
        torch.save(state, './checkpoint_sens/ckpt_%d.t7' % epoch)
        torch.save(state, './checkpoint_sens/ckpt.t7' )
        best_acc = acc

    return test_loss/len(testloader), 100.*correct/total, score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch nEXO background rejection')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--config', '-f', type=str, default="baseline.yml", help="specify yaml config")

    args = parser.parse_args()
    # parameters
    config = yaml_load(args.config)
    model_name = config['model']['name'] 
    data_name = config['data']['name']
    h5file = config['data']['h5name']
    fcsv = config['data']['csv']
    input_shape = [int(i) for i in config['data']['input_shape']]
    lr = config['fit']['compile']['initial_lr']
    batch_size = config['fit']['batch_size']
    epochs = config['fit']['epochs']

    # Data
    print('==> Preparing data..')
    nEXODataset = DenseDataset(h5file, fcsv, n_channels=input_shape[2])
    # Creating data indices for training and validation splits:
    dataset_size = len(nEXODataset)
    indices = list(range(dataset_size))
    validation_split = .2
    split = int(np.floor(validation_split * dataset_size))
    shuffle_dataset = True
    random_seed= 40
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(nEXODataset, batch_size=batch_size, sampler=train_sampler, num_workers=8)
    validation_loader = torch.utils.data.DataLoader(nEXODataset, batch_size=batch_size, sampler=validation_sampler, num_workers=8)

    start_epoch = 0

    print('==> Building model..')
    if model_name == 'senet':
        net = resnet.resnet18(num_classes=2, use_senet=True, ratio=16)
    elif model_name == 'resnet18':
        net = resnet18(input_channels=input_shape[2])
    elif model_name == 'resnet14':
        net = resnet14(input_channels=input_shape[2])
    else:
        print('Model {} not defined!'.format(model_name))
        raise SystemExit()
    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # We use SGD
    # optimizer = torch.optim.SGD(net.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999),
                                  weight_decay=1e-4, eps=1e-08, amsgrad=False)

    net = net.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use ", torch.cuda.device_count(), " GPUs!")
        net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())))

    if args.resume and os.path.exists('./checkpoint_sens/ckpt.t7'):
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint_sens'), 'Error: no checkpoint directory found!'
        if device == 'cuda':
            checkpoint = torch.load('./checkpoint_sens/ckpt.t7' )
        else:
            checkpoint = torch.load('./checkpoint_sens/ckpt.t7', map_location=torch.device('cpu') )
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1

    # Numpy arrays for loss and accuracy, if resume from check point then read the previous results
    if args.resume and os.path.exists('./training_outputs/loss_acc.npy'):
        arrays_resumed = np.load('./training_outputs/loss_acc.npy', allow_pickle=True)
        y_train_loss = arrays_resumed[0]
        y_train_acc  = arrays_resumed[1]
        y_valid_loss = arrays_resumed[2]
        y_valid_acc  = arrays_resumed[3]
        test_score   = arrays_resumed[4].tolist()
    else:
        y_train_loss = np.array([])
        y_train_acc  = np.array([])
        y_valid_loss = np.array([])
        y_valid_acc  = np.array([])
        test_score   = []

    for epoch in range(start_epoch, start_epoch + epochs):
        # Set the learning rate
        adjust_learning_rate(optimizer, epoch, lr)
        iterout = "\nEpoch [%d]: "%(epoch)

        for param_group in optimizer.param_groups:
            iterout += "lr=%.3e"%(param_group['lr'])
            print(iterout)
            try:
                train_ave_loss, train_ave_acc = train(train_loader, epoch)
            except Exception as e:
                print("Error in training routine!")
                print(e.message)
                print(e.__class__.__name__)
                traceback.print_exc(e)
                break
            print("Train[%d]: Result* Loss %.3f\t Accuracy: %.3f"%(epoch, train_ave_loss, train_ave_acc))
            y_train_loss = np.append(y_train_loss, train_ave_loss)
            y_train_acc = np.append(y_train_acc, train_ave_acc)

            # Evaluate on validationset
            try:
                valid_loss, prec1, valid_score= test(validation_loader, epoch, True)
            except Exception as e:
                print("Error in validation routine!")
                print(e.message)
                print(e.__class__.__name__)
                traceback.print_exc(e)
                break
            print("Test[%d]: Result* Loss %.3f\t Precision: %.3f"%(epoch, valid_loss, prec1))


            test_score.append(valid_score)
            y_valid_loss = np.append(y_valid_loss, valid_loss)
            y_valid_acc = np.append(y_valid_acc, prec1)

            np.save('./training_outputs/loss_acc.npy', np.array([y_train_loss, y_train_acc, y_valid_loss, y_valid_acc, test_score], dtype=object))
