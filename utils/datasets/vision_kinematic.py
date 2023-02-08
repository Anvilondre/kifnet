import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

from utils.datasets import shift

def get_df(path: pd.DataFrame):
    
    ''' Returns a dataframe with absolute image paths '''
    
    df = pd.read_csv(path).reset_index(drop=True)
    
    fldr = '/'.join(path.split('/')[:-2])+'/frames/'
    df['full_frames'] = fldr+df['frames']
    
    return df.sort_values('relTime').reset_index(drop=True)

class Vision_Lags2One(Dataset):
    
    """
    Dataset class for creating Lags -> One (in terms of time steps) mappings
    Contains images and kinematic features
    """
    
    def __init__(self,
                 data: pd.DataFrame,
                 lags: list,
                 offset: int,
                 stride: int,
                 resize: tuple,
                 ds_mean: list,
                 ds_std: list,
                 config: dict) -> None:
        """
        Args:
            data: dataset with image pathes and kinematic features
            lags: dataset with image pathes and kinematic features
            offset: Offset before the @y value (0 means it will be the next value after @X)
            stride: Difference in time steps between two generated Input -> Output sequences
            resize: The size of the resized image
            ds_mean: Dataset mean values for normalization
            ds_std: Dataset std values for normalization
        Returns:
            None
        """
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
        """
        Args:
            idx: index of a sample, same as index of the first element in returned Input sequence
        Returns: 
            X, y, im: Input sequence of kinematic features, One target vector and one image respectively
        """
        img = self.frames[idx * self.stride]
        cv = self.transform(Image.open(img))
        
        return self.X[idx * self.stride],\
               self.y[idx * self.stride + self.offset],\
               cv
