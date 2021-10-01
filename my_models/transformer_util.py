import os
from torch.utils.data import Dataset

class ArxivDataset(Dataset):
    def __init__(self, data_path, split):
        self.samples = []
        
        train_split = 'train_small.txt'
        val_split = 'val_small.txt'
        test_split = 'test_small.txt'
        X = []
        y = []
        if split == 'train':
            with open(os.path.join(data_path, 'X_'+train_split), encoding = 'utf-8') as f:
                X = f.readlines()
            with open(os.path.join(data_path, 'y_'+train_split), encoding = 'utf-8') as f:
                y = f.readlines()
        elif split == 'val':
            with open(os.path.join(data_path, 'X_'+val_split), encoding = 'utf-8') as f:
                X = f.readlines()
            with open(os.path.join(data_path, 'y_'+val_split), encoding = 'utf-8') as f:
                y = f.readlines()
        elif split == 'test':
            with open(os.path.join(data_path, 'X_'+test_split), encoding = 'utf-8') as f:
                X = f.readlines()
            with open(os.path.join(data_path, 'y_'+test_split), encoding = 'utf-8') as f:
                y = f.readlines()
        else:
            raise Exception('Dataset Not Found!')
        
        if split in ['train', 'val', 'test']:
            self.samples = list(zip(X, y))

        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]