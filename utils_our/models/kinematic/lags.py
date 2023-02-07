import torch
from torch import nn
import torch.nn.functional as F
from typing import List

class LinearBlock(nn.Module):

    ''' Linear -> LeakyReLU '''
    
    def __init__(self, inp_dim: int, out_dim: int, slope: float = -0.01):
        super(LinearBlock, self).__init__()
        
        self.fc = nn.Linear(inp_dim, out_dim)
        self.relu = nn.LeakyReLU(slope)
        
        
    def forward(self, X):
        return self.relu(self.fc(X))
    
    
class Single_Single(nn.Module):
    
    ''' Sequence of SimpleBlocks -> Linear '''
    
    def __init__(self, inp_dim: int, out_dim: int, hidden_dims: List[int]):
        super(Single_Single, self).__init__()
        
        self.model = nn.Sequential(
            LinearBlock(inp_dim, hidden_dims[0]),
            *[LinearBlock(i, o) for i, o in zip(hidden_dims, hidden_dims[1:])],
            nn.Linear(hidden_dims[-1], out_dim),
        )
        
    def forward(self, X):
        return self.model(X)
    
    
class Single_Single_Separate(nn.Module):
    
    ''' Sequence of SimpleBlocks -> (Sequence of SimpleBlocks + Linear) for every output '''
    
    def __init__(
        self,
        inp_dim: int,
        out_dim: int,
        encoder_dims: List[int],
        decoder_dims: List[int],
        device: str
    ):
        
        super(Single_Single_Separate, self).__init__()
        
        self.encoder = nn.Sequential(
            SimpleBlock(inp_dim, encoder_dims[0]),
            *[SimpleBlock(i, o) for i, o in zip(encoder_dims, encoder_dims[1:])]
        ).to(device)
        
        self.decoders = [nn.Sequential(
            SimpleBlock(encoder_dims[-1], decoder_dims[0]),
            *[SimpleBlock(i, o) for i, o in zip(decoder_dims, decoder_dims[1:])],
            nn.Linear(decoder_dims[-1], 1)
        ).to(device) for _ in range(out_dim)]
    
    def forward(self, X):
        enc = self.encoder(X)
        
        # outs = [dec(enc) for dec in self.decoders]
        outs = torch.cat([dec(enc) for dec in self.decoders], dim=1)
        
        return outs
        
        