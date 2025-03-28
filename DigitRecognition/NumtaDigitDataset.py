import torch
import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt
import DataProcessing as DP

class NumtaDigitDataset(torch.utils.data.Dataset): # CustomDataset
    def __init__(self, datapath, data, data_st, invalids = []):
        self.datapath = datapath
        self.X, self.T = (data[0], data[1]) if isinstance(data, tuple) else (data, None)
        self.data_st = data_st
        self.classes = np.unique(self.T) if self.T is not None else None
        self.invalids = invalids
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        #feature = plt.imread(self.datapath + '/' + self.X[idx]).swapaxes(0, 2)
        if self.X[idx] in self.invalids or np.array(Image.open(self.datapath + '/' + self.X[idx])).ndim < 2:
            idx = np.random.randint(0, len(self)-1)
        feature = DP.to_gray(DP.resize(self.datapath + '/' + self.X[idx]))
        #if feature.ndim != 3:
        #    return
        feature = feature.transpose(2, 0, 1)
        feature_stdzd = self.transform(feature, self.data_st._standardizeX)
        if self.T is None:
            return feature_stdzd.float() 
        else:
            target = self.T[idx]
            Tn = np.where(target == self.classes)[0]
            return feature_stdzd.float(), torch.tensor(Tn)
    
    def transform(self, sample, st_func):
        return torch.tensor(st_func(sample))
