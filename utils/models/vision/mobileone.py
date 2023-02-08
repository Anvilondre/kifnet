import sys
import torch
from torch import nn

sys.path.append('../ml-mobileone')

from mobileone import mobileone

class VisionModel(nn.Module):
    
    '''Wrapper around MobileOne with the custom output size'''
    
    def __init__(self, embedding_dim: int, mobone_type: str, mobone_path: str):
        super(VisionModel, self).__init__()

        self.backbone = mobileone(variant=mobone_type)
        self.checkpoint = torch.load(mobone_path)
        self.backbone.load_state_dict(self.checkpoint)
        self.num_ftrs = self.backbone.linear.in_features
        self.backbone.linear = nn.Linear(self.num_ftrs, embedding_dim)
        
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.backbone.gap.requires_grad = True
        self.backbone.linear.requires_grad = True
        
    def forward(self, x):
        return self.backbone(x)