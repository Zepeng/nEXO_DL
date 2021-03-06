import torch, h5py
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import pandas as pd

class H5Dataset(data.Dataset):
    def __init__(self, h5_path, csv_path, n_channels=2):
        self.to_tensor = transforms.ToTensor()
        csv_info = pd.read_csv(csv_path, header=None)
        self.groupname = np.asarray(csv_info.iloc[:,0])
        self.datainfo = np.asarray(csv_info.iloc[:,1])
        self.h5file = h5py.File(h5_path, 'r')
        self.n_channels = n_channels

    def __len__(self):
        return len(self.datainfo)

    def __getitem__(self, idx):
        dset_entry = self.h5file[self.groupname[idx]][self.datainfo[idx]]
        eventtype = dset_entry.attrs[u'tag']
        img = np.array(dset_entry)[:,:,:self.n_channels]
        img = np.transpose(img, (2,0,1)) #the initial image building put the layer index at axe 3.
        return torch.from_numpy(img).type(torch.FloatTensor), eventtype


def test():
    dataset = H5Dataset('test.h5', 'test.csv')
    print(1, dataset[1][0].shape)
    print(1, dataset[1][1].shape)

if __name__ == '__main__':
    test()
