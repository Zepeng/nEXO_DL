#!/usr/bin/env python
# coding: utf-8

#Dataset code copied from https://github.com/utkuozbulak/pytorch-custom-dataset-examples
#model code copied from https://github.com/DeepLearnPhysics/pytorch-uresnet
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import os

import torch.optim as optim
from torch.optim import lr_scheduler

import argparse
from networks.resnet_regression import resnet18
from utils.data_loaders import VertexDataset
from utils.data_loaders import DatasetFromSparseMatrix
import traceback
import yaml 

valid_loss_min = np.inf

def yaml_load(config):
    with open(config) as stream:
        param = yaml.safe_load(stream)
    return param

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
list_of_gpus = range(torch.cuda.device_count())
device_ids = range(torch.cuda.device_count())
os.environ["CUDA_VISIBLE_DEVICES"] = str(list_of_gpus).strip('[]').replace(' ', '')

start_epoch = 0  # start from epoch 0 or last checkpoint epoch

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = lr/np.exp(epoch/10)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Training
def train(trainloader, epoch):
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        print(batch_idx, '/', len(trainloader), 'Loss: %.3f ' % (train_loss/(batch_idx+1.)))
    return train_loss/len(trainloader) 

def test(testloader, epoch):
    global valid_loss_min
    net.eval()
    test_loss = 0
    score = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            for target, output in zip(targets, outputs):
                score.append([target.detach().cpu().numpy(), output.detach().cpu().numpy()])
            print(batch_idx, '/', len(testloader), 'Loss: %.3f ' % (test_loss/(batch_idx+1.)))

    if test_loss < valid_loss_min:
        valid_loss_min = test_loss
        print ('Saving...')
        state = {
            'net': net.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint_sens'):
            os.mkdir('checkpoint_sens' )
        if not os.path.isdir('training_outputs'):
            os.mkdir('training_outputs')
        torch.save(state, './checkpoint_sens/ckpt_%d.t7' % epoch)
        
    return test_loss/len(testloader), np.array(score) #, is_better

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch nEXO background rejection')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--config', '-f', type=str, default="baseline.yml", help="specify yaml config")
    args = parser.parse_args()
    # parameters
    config = yaml_load(args.config)
    data_name = config['data']['name']
    h5file = config['data']['h5name']
    fcsv = config['data']['csv']
    input_shape = [int(i) for i in config['data']['input_shape']]
    lr = config['fit']['compile']['initial_lr']
    batch_size = config['fit']['batch_size']
    epochs = config['fit']['epochs']

    # Data
    print('==> Preparing data..')
    nEXODataset = VertexDataset(h5file, fcsv, n_channels=input_shape[2])
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
    train_loader = torch.utils.data.DataLoader(nEXODataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    validation_loader = torch.utils.data.DataLoader(nEXODataset, batch_size=batch_size, sampler=validation_sampler, num_workers=0)

    start_epoch = 0

    print('==> Building model..')
    net = resnet18(input_channels=input_shape[2])
    # Define loss function (criterion) and optimizer
    criterion = nn.MSELoss().cuda()
    # We use SGD
    # optimizer = torch.optim.SGD(net.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), 
                                  weight_decay=1e-4, eps=1e-08, amsgrad=False)

    net = net.to(device)
    
    if torch.cuda.device_count() > 1:
        print("Let's use ", torch.cuda.device_count(), " GPUs!")
        net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())))
        
    if args.resume and os.path.exists('./checkpoint_sens/ckpt_49.t7'):
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint_sens'), 'Error: no checkpoint directory found!'
        if device == 'cuda':
            checkpoint = torch.load('./checkpoint_sens/ckpt_49.t7' )
        else:
            checkpoint = torch.load('./checkpoint_sens/ckpt_49.t7', map_location=torch.device('cpu') )
        net.load_state_dict(checkpoint['net'])
        valid_loss_min = checkpoint['loss']
        start_epoch = checkpoint['epoch'] + 1
        
    # Numpy arrays for loss, if resume from check point then read the previous results
    if args.resume and os.path.exists('./training_outputs/loss_acc.npy'):
        arrays_resumed = np.load('./training_outputs/loss_acc.npy', allow_pickle=True)
        y_train_loss = arrays_resumed[0]
        y_valid_loss = arrays_resumed[1]
        test_score   = arrays_resumed[2].tolist()
    else:
        y_train_loss = np.array([])
        y_valid_loss = np.array([])
        test_score   = []
    
    for epoch in range(start_epoch, start_epoch + epochs):
        # Set the learning rate
        adjust_learning_rate(optimizer, epoch, lr)
        iterout = "\nEpoch [%d]: "%(epoch)
        
        for param_group in optimizer.param_groups:
            iterout += "lr=%.3e"%(param_group['lr'])
            print(iterout)
            try:
                train_ave_loss = train(train_loader, epoch)
            except Exception as e:
                print("Error in training routine!")
                print(e.message)
                print(e.__class__.__name__)
                traceback.print_exc(e)
                break
            print("Train[%d]: Result* Loss %.3f\t "%(epoch, train_ave_loss))
            y_train_loss = np.append(y_train_loss, train_ave_loss)

            # Evaluate on validationset
            try:
                valid_loss, score = test(validation_loader, epoch)
            except Exception as e:
                print("Error in validation routine!")
                print(e.message)
                print(e.__class__.__name__)
                traceback.print_exc(e)
                break
                
            print("Test[%d]: Result* Loss %.3f\t "%(epoch, valid_loss))
            
            test_score.append(score)
            y_valid_loss = np.append(y_valid_loss, valid_loss)
            
            np.save('./training_outputs/loss_acc.npy', np.array([y_train_loss, y_valid_loss, test_score], dtype=object))
