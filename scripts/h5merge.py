#!/usr/bin/env python
'''Merge hdf5 files into a single file'''
import h5py    # HDF5 support
import glob
import time
import csv
import argparse
import pandas as pd
from networks.resnet_example import resnet18
import numpy as np
import torch
net = resnet18(input_channels=2)

def h5merger(filedir, c_file, h5_file):
    filelist = glob.glob('%s/*.h5' % filedir)
    csvfile = open(c_file, 'w')
    fieldnames = ['groupname', 'dsetname']
    writer = csv.DictWriter(csvfile, fieldnames)
    nsig = 0
    nbkg = 0
    with h5py.File(h5_file, 'w') as fid:
        for i in range(len(filelist)):
            fileName = filelist[i]
            groupname = 'nexo_data'
            f = h5py.File(fileName,  "r")
            f.copy(f[groupname], fid['/'], name='nexo_data_%d' % i)
            dset = f[groupname]
            if 'bb0n' in fileName:
                if nsig > 400000:
                    continue
                nsig += len(dset.keys())
            else:
                if nbkg > 400000:
                    continue
                nbkg += len(dset.keys())
            print(nsig, nbkg, fileName)
            for item in dset.keys():
                img = np.array(dset[item])[:,:,:2]
                if np.isnan(img).any() or np.isinf(img).any():
                    print('NAN', item)
                    continue
                else:
                #img = np.transpose(img, (2,0,1)) #the initial image building put the layer index at axe 3.
                #target = np.zeros((1, 2, 255, 255)) #the initial image building put the layer index at axe 3.
                #target[0,:, :200, :] = img #the initial image building put the layer index at axe 3.
                #if not torch.isnan(net(torch.from_numpy(target).type(torch.FloatTensor)))[0,0].item():
                    writer.writerow({'groupname':'nexo_data_%d' % i, 'dsetname':item})
            f.close()
    print('Signal:', nsig, 'Background:', nbkg)
    csvfile.close()
    csv_info = pd.read_csv(c_file, header=None, delimiter=',')
    shuffled = csv_info.sample(frac=1).reset_index()
    n_train = int(len(shuffled)*0.8)
    shuffled = csv_info.sample(frac=1).reset_index()
    shuffled[:n_train].to_csv(c_file.replace('nexo', 'nexo_train'), index=False, columns =[0, 1])
    shuffled[n_train:].to_csv(c_file.replace('nexo', 'nexo_valid'), index=False, columns =[0, 1])
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dataset builder.')
    parser.add_argument('--filedir', '-f', type=str, help='directory of h5 files.')
    parser.add_argument('--outfile', '-o', type=str, help='output h5 file.')
    parser.add_argument('--csvfile', '-c', type=str, help='csv file of dataset info.')
    args = parser.parse_args()
    h5merger(args.filedir, args.csvfile, args.outfile)
