import torch
import abc 

class Scaler(metaclass=abc.ABCMeta):
    def __init__(self):
        # Name for printing
        self._name = None
        
    def transform(self, x):
        raise NotImplementedError
        
    def inverse_transform(self, x):
        raise NotImplementedError
    
class StandardScaler(Scaler):
    def __init__(self, mean, std):
        self._name = 'StandardScaler'
        self.mean = mean
        self.std = std
        
    def transform(self, x):
        return (x - self.mean) / self.std
    
    def inverse_transform(self, x):
        return (x * self.std) + self.mean
        
class MinMaxScaler(Scaler):
    def __init__(self, minimum, maximum):
        self._name = 'MinMaxScaler'
        self.min = minimum
        self.max = maximum
        
    def transform(self, x):
        return (x - self.min) / (self.max - self.min)
    
    def inverse_transform(self, x):
        return x * (self.max - self.min) + self.min
    
    
class IdScaler(Scaler):
    def __init__(self, *args, **kwargs):
        self._name = 'IdScaler'
        
    def transform(self, x):
        return x
    
    def inverse_transform(self, x):
        return x