import torch
from torch.utils.data import Dataset
from typing import List

class Kinematic_Seq2Seq(Dataset):
    
    """
    Dataset class for creating Sequence -> Sequence mappings
    """
    
    def __init__(self, X, y, x_len: int, y_len: int, stride: int = 1) -> None:
        """
        Args:
            X: Features time-sorted in ascending order (may be anything convertable to torch.FloatTensor)
            y: Target values, must be aligned with @X (may be anything convertable to torch.FloatTensor)
            x_len: Length of the input sequence
            y_len: Length of the output sequence
            stride: Difference in time steps between two generated Input -> Output sequences
        Returns:
            None
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.x_len = x_len
        self.y_len = y_len
        self.stride = stride
        
        # Total number of Input -> Output sequnces that will be generated
        self.num_samples = (len(self.X) - x_len - y_len + 1) // stride
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Args:
            idx: index of a seq2seq sample, same as index of the first element in returned Input sequence
        Returns: 
            X, y: Input and Output sequences respectively, with no intersection
        """
        return self.X[idx * self.stride: idx * self.stride + self.x_len],\
               self.y[idx * self.stride + self.x_len: idx * self.stride + self.x_len + self.y_len]

class Kinematic_Seq2One(Dataset):
    
    """
    Dataset class for creating Sequence -> One (in terms of time steps) mappings
    """
    
    def __init__(self, X, y, x_len: int, offset: int = 0, stride: int = 1) -> None:
        """
        Args:
            X: Features time-sorted in ascending order
            y: Target values, must be aligned with @X
            x_len: Length of the input sequence
            offset: Offset before the @y value (0 means it will be the next value after @X)
            stride: Difference in time steps between two generated Input -> One items
        Returns:
            None
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.x_len = x_len
        self.offset = offset
        self.stride = stride
        
        # Total number of sequnces that will be generated
        self.num_samples = (len(self.X) - x_len - offset) // stride
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Args:
            idx: index of a sample, same as index of the first element in returned Input sequence
        Returns: 
            X, y: Input sequence and One target vector respectively
        """
        return self.X[idx * self.stride: idx * self.stride + self.x_len],\
               self.y[idx * self.stride + self.x_len + self.offset]
    
def shift(xs: torch.Tensor, n: int):
    """
    Shifts a tensor by n steps, and fills the rest with nans
    Args:
        xs: Tensor to be shifted
    Returs:
        Shifted tensor !! Contains NaNs if @n != 0
    """
    if n == 0:
        return xs
    elif n > 0:
        return torch.cat(
            [ # Concatenating the nan beginning with the shifted end
                torch.full((n, xs.shape[1]), torch.nan),
                xs[:-n, :]
            ], dim=0
        )
    else:
        return torch.cat(
            [ # Concatenating the nan end with the shifted beginning
                xs[-n:, :],
                torch.full((-n, xs.shape[1]), torch.nan)
            ], dim=0
        )
    
class Kinematic_Lags2One(Dataset):
    
    """
    Dataset class for creating Lags -> One (in terms of time steps) mappings
    Drops first @max_lag values to not include NaNs
    !! Caution: was not tested on negative lags and is only intended to be used with non-negative ones
    """
    
    def __init__(self, X, y, lags: List[int], offset: int = 0, stride: int = 1) -> None:
        """
        Args:
            X: Features time-sorted in ascending order
            y: Target values, must be aligned with @X
            lags: Lags to be calculated on X, !! Only use non-negative lags
            offset: Offset before the @y value (0 means it will be the next value after @X)
            stride: Difference in time steps between two generated Input -> One items
        Returns:
            None
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.max_lag = max(lags)
        self.offset = offset
        self.stride = stride
        
        max_lag = max(lags)
        self.X = torch.cat([shift(self.X, lag) for lag in lags], dim=1)[max_lag:]
        self.y = self.y[max_lag:]
        
        # Total number of sequnces that will be generated
        self.num_samples = (len(self.X) - self.offset) // self.stride
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Args:
            idx: index of a sample, same as index of the first element in returned Input sequence
        Returns: 
            X, y: Input sequence and One target vector respectively
        """
        return self.X[idx * self.stride],\
               self.y[idx * self.stride + self.offset]
    
class Kinematic_Lags2Residual(Dataset):
    
    """
    Dataset class for creating Lags -> One Residual (in terms of time steps) mappings
    Drops first @max_lag values to not include NaNs
    !! Caution: was not tested on negative lags and is only intended to be used with non-negative ones
    """
    
    def __init__(self, X, y, lags: List[int], residual: int, offset: int = 0, stride: int = 1) -> None:
        """
        Args:
            X: Features time-sorted in ascending order
            y: Target values, must be aligned with @X
            residual: Residual lag
            lags: Lags to be calculated on X, !! Only use non-negative lags
            offset: Offset before the @y value (0 means it will be the next value after @X)
            stride: Difference in time steps between two generated Input -> One items
        Returns:
            None
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.max_lag = max(max(lags), residual)
        self.offset = offset
        self.stride = stride
        
        max_lag = max(lags)
        self.residual_lag = shift(self.X, residual)[max_lag:]
        self.X = torch.cat([shift(self.X, lag) for lag in lags], dim=1)[max_lag:]
        self.y = self.y[max_lag:]
        
        # Total number of sequnces that will be generated
        self.num_samples = (len(self.X) - self.offset) // self.stride
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Args:
            idx: index of a sample, same as index of the first element in returned Input sequence
        Returns: 
            X, y: Input sequence and One target vector respectively
        """
        return self.X[idx * self.stride],\
               self.y[idx * self.stride + self.offset] - self.residual_lag[idx * self.stride],\
               self.residual_lag[idx * self.stride]