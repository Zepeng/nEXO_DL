import numpy  as np
import pandas as pd
import torch, h5py
import torch.utils.data as data
from torchvision import transforms
from scipy.sparse import csr_matrix

class SparseData(torch.utils.data.Dataset):
    def __init__(self, csv_path, h5path, nevents=None, augmentation = False):
        """ This class yields events from pregenerated MC file.
        Parameters:
            csv_path : str; csv file containing file/event information to read
            h5path : str; name of H5 data file 
        """
        self.csv_info = pd.read_csv(csv_path, skiprows=1 , header=None)
        self.groupname = np.asarray(self.csv_info.iloc[:,0])
        self.datainfo = np.asarray(self.csv_info.iloc[:,1])
        self.h5file = h5py.File(h5path, 'r')
        self.augmentation = augmentation

    def __getitem__(self, idx):
        dset_entry = self.h5file[self.groupname[idx]][self.datainfo[idx]]
        eventtype = dset_entry.attrs[u'tag']
        label = 1 
        if eventtype == 'gamma':
            label = 0 
        return (dset_entry[:, 0]).astype(int) + 120, (dset_entry[:, 1]).astype(int) + 120, (dset_entry[:, 2]).astype(int) + 20, dset_entry[:, 3]/10000, np.unique([label]), idx

    def __len__(self):
        return len(self.datainfo)
    def __del__(self):
        if self.h5file is not None:
            self.h5file.close()
        if self.csv_info is not None:
            del self.csv_info

def collatefn(batch):
    coords = []
    energs = []
    labels = []
    events = np.zeros(len(batch))
    for bid, data in enumerate(batch):
        x, y, z, E, lab, event = data
        batchid = np.ones_like(x)*bid
        coords.append(np.concatenate([x[:, None], y[:, None], z[:, None], batchid[:, None]], axis=1))
        energs.append(E)
        labels.append(lab)
        events[bid] = event

    coords = torch.tensor(np.concatenate(coords, axis=0), dtype = torch.short)
    energs = torch.tensor(np.concatenate(energs, axis=0), dtype = torch.float).unsqueeze(-1)
    labels = torch.tensor(np.concatenate(labels, axis=0), dtype = torch.long)

    return  coords, energs, labels, events

class DenseDataset(data.Dataset):
    def __init__(self, h5_path, csv_path, n_channels=2):
        """ This class yields events from pregenerated MC file.
        Parameters:
            csv_path : str; csv file containing file/event information to read
            h5path : str; name of H5 data file 
        """
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

class DatasetFromSparseMatrix(data.Dataset):
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
        h5_entry = self.h5file[self.groupname[idx]][self.datainfo[idx]]
        dset_entry_x = csr_matrix((h5_entry['data_x'][:], h5_entry['indices_x'][:],
                                   h5_entry['indptr_x'][:]), h5_entry.attrs['shape_x'], dtype=np.float32).todense()
        dset_entry_y = csr_matrix((h5_entry['data_y'][:], h5_entry['indices_y'][:],
                                   h5_entry['indptr_y'][:]), h5_entry.attrs['shape_y'], dtype=np.float32).todense() 
        eventtype = 1 if h5_entry.attrs[u'tag']=='e-' else 0
        img = np.array([dset_entry_x, dset_entry_y])
        return torch.from_numpy(img).type(torch.FloatTensor), eventtype
    
def test():
    dataset = DenseDataset('test.h5', 'test.csv')
    print(1, dataset[1][0].shape)
    print(1, dataset[1][1].shape)

if __name__ == '__main__':
    test()
