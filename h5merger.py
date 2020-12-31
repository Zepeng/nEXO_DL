#!/usr/bin/env python
'''Reads NeXus HDF5 files using h5py and prints the contents'''

import h5py    # HDF5 support
import glob
import time
import csv
filelist = glob.glob('/scratchfs/exo/zepengli94/nexo/*.h5')
t1 = time.time()
csvfile = open('dataset_info.csv', 'w')
fieldnames = ['groupname', 'dsetname']
writer = csv.DictWriter(csvfile, fieldnames)
with h5py.File('test1.h5', 'w') as fid:
    junodata = fid.create_group(u'nexo_data' )
    for i in range(len(filelist)):
        fileName = filelist[i]
        print(fileName, time.time() - t1)
        f = h5py.File(fileName,  "r")
        f.copy(f['bb0n.tar'], junodata, name='nexo_data_%d' % i)
        dset = f['bb0n.tar']
        for item in dset.keys():
            writer.writerow({'groupname':'nexo_data_%d' % i, 'dsetname':item})
        f.close()
