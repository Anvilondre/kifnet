import numpy as np
from sklearn.metrics import mean_squared_error
import torch
from typing import Tuple
import abc 

class Metric(metaclass=abc.ABCMeta):
    def __init__(self):
        # Name for printing
        self._name = None
        
    def __call__(self, y_pred, y):
        """
        Function to compute the metric score
        
        Args:
            y_pred: Predictions
            y: True values
        
        Returns: Metric value
        """
        raise NotImplementedError
        
    def __str__(self):
        return f'Metric({self._name})'
    

def unscale_metric(scaler):
    """
    Decorator that can be applied to any Metric class with an appropriate sklearn-style scaler
    """
    def wrap(cls):
        init_call = cls.__call__
        def new_call(self, Yhat, Y):
            return init_call(self,
                            scaler.inverse_transform(Yhat),
                            scaler.inverse_transform(Y))
    
        cls.__call__ = new_call
        return cls
    
    return wrap


class MAPE(Metric):
    def __init__(self):
        self._name = 'mape'

    def __call__(self, y_pred, y):
        return torch.mean(torch.absolute((y - y_pred) / y))


class WAPE(Metric):
    def __init__(self):
        self._name = 'wape'

    def __call__(self, y_pred, y):
        return torch.sum(torch.absolute(y - y_pred)) / torch.sum(torch.absolute(y))

class MAE(Metric):
    def __init__(self):
        self._name = 'mae'
            
    def __call__(self, y_pred, y):
        return torch.mean(torch.absolute(y - y_pred))

class RMSE(Metric):
    def __init__(self):
        self._name = 'rmse'
        
    def __call__(self, y_pred, y):
        return torch.sqrt(torch.mean(torch.square(y - y_pred)))

class MSE(Metric):
    def __init__(self):
        self._name = 'mse'
        
    def __call__(self, y_pred, y):
        return torch.mean(torch.square(y - y_pred))
    
def split_rmse(y_pred: torch.Tensor, y: torch.Tensor, split: int) -> Tuple[int, int]:
    """
    Function to calculate RMSE on non-overlaping windows of size @split
    
    Args:
        y_pred: Tensor with predictions
        y: Tensor with ground truth values (should be the same length as @y_pred)
        split: Split window size
    Returns:
        mean, std of per-window metrics
    """
    split_y = y.split(split)
    split_pred = y_pred.split(split)
    rmse = RMSE()
    split_metric = [rmse(yhat, y) for y, yhat in zip(split_y, split_pred)]
    
    return np.mean(split_metric), np.std(split_metric)


def split_mae(y_pred: torch.Tensor, y: torch.Tensor, split: int) -> Tuple[int, int]:
    """
    Function to calculate MAE on non-overlaping windows of size @split
    
    Args:
        y_pred: Tensor with predictions
        y: Tensor with ground truth values (should be the same length as @y_pred)
        split: Split window size
    Returns:
        mean, std of per-window metrics
    """
    split_y = y.split(split)
    split_pred = y_pred.split(split)
    mae = MAE()
    split_metric = [mae(yhat, y) for y, yhat in zip(split_y, split_pred)]
    
    return np.mean(split_metric), np.std(split_metric)