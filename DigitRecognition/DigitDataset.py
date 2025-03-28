import torch
import numpy as np

class DigitDataset(torch.utils.data.Dataset): # CustomDataset
    def __init__(self, data, data_st):
        self.X = data[0]
        self.T = data[1]
        self.data_st = data_st
        self.classes = np.unique(self.T)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        feature = self.X[idx]
        target = self.T[idx]
        feature_stdzd = self.transform(feature, self.data_st._standardizeX)
        Tn = np.where(target == self.classes)[0]
        return feature_stdzd.float(), torch.tensor(Tn)
    
    def transform(self, sample, st_func):
        return torch.tensor(st_func(sample))
