import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

from utils_our.datasets import shift

def get_df(path):
    df = pd.read_csv(path).reset_index(drop=True)
    fldr = '/'.join(path.split('/')[:-2])+'/frames/'
    df['full_frames'] = fldr+df['frames']
    return df.sort_values('relTime').reset_index(drop=True)

class Vision_Lags2One(Dataset):
    
    def __init__(self, data, lags,
                 offset: int, stride: int,
                 resize, ds_mean, ds_std,
                 config) -> None:
        
        self.data = data
        self.X = torch.FloatTensor(self.data[config['features']].values)
        self.y = torch.FloatTensor(self.data[config['targets']].values)
        self.max_lag = max(lags)
        self.offset = offset
        self.stride = stride
        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(ds_mean, ds_std),
        ])
        self.max_lag = max(lags)
        
        self.frames = self.data['full_frames'].tolist()[self.max_lag:]
        self.X = torch.cat([shift(self.X, lag) for lag in lags], dim=1)[self.max_lag:]
        self.y = self.y[self.max_lag:]
        
        # Total number of sequnces that will be generated
        self.num_samples = (len(self.X) - self.offset) // self.stride
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx: int) -> tuple:
        img = self.frames[idx * self.stride]
        cv = self.transform(Image.open(img))
        
        return self.X[idx * self.stride],\
               self.y[idx * self.stride + self.offset],\
               cv