import torch
from torch import nn
from utils.models import Single_Single, VisionModel
from typing import List

def ada_in(style: torch.Tensor, content: torch.Tensor, ax=1):
    
    """
    Implementation of Adaptive Instance Normalization
    between style and content vectors
    """
    
    style_std = style.std(axis=ax).view(-1,1)
    content_std = content.std(axis=ax).view(-1,1)
    
    style_mean = style.mean(axis=ax).view(-1,1)
    content_mean = content.mean(axis=ax).view(-1,1)
    
    t = style_std * (content - content_mean) / content_std + style_mean
    
    return t


class FusionNet(nn.Module):
    
    """
    A part of KIFNet to fuse kinematic and image vectors
    """
        
    def __init__(self, fusion: str, fusion_dims: List[int], decoder_output: int):
        """
        Args:
            fusion: type of fusion ('avg', 'max',
                                    'concat', 'concat-ln', 'concat-bn',
                                    'adain-kin-cv', 'adain-cv-kin',
                                    'dot-att', 'mlp-att')
            fusion_dims: the number of neurons in each MLP layer 
            decoder_output: output size of the fused vector
        """
        super(FusionNet, self).__init__()
        self.sigmoid_net = Single_Single(inp_dim=decoder_output*2,
                                         out_dim=1,
                                         hidden_dims=fusion_dims)

        self.bn = nn.BatchNorm1d(decoder_output)
        self.ln = nn.LayerNorm(decoder_output)
        self.fusion = fusion
        
    def forward(self, kin, cv):

        if self.fusion == 'avg':
            fusion_out = (cv + kin) / 2

        if self.fusion == 'max':
            fusion_out = torch.max(
                torch.stack([kin, cv], axis=0),
                axis=0
            )[0]

        if self.fusion.startswith('concat'):
            fusion_out = torch.cat([cv, kin], axis=1)

        if self.fusion == 'concat-ln':
            fusion_out = self.ln(fusion_out)

        if self.fusion == 'concat-bn':
            fusion_out = self.bn(fusion_out)

        if self.fusion == 'adain-kin-cv':
            fusion_out = ada_in(kin, cv)

        if self.fusion == 'adain-cv-kin':
            fusion_out = ada_in(cv, kin)

        if self.fusion == 'dot-att':
            concat_temp = torch.cat([cv, kin], axis=1)
            alpha = (cv * kin).sum(axis=1)
            alpha = torch.unsqueeze(alpha, 1)
            fusion_out = cv * alpha + kin * (1 - alpha)

        if self.fusion == 'mlp-att':
            concat_temp = torch.cat([cv, kin], axis=1)
            alpha = self.sigmoid_net(concat_temp)
            fusion_out = cv * alpha + kin * (1 - alpha)
            
        return fusion_out
    

class KIFNet(nn.Module):
    
    ''' KIFNet model '''
    
    def __init__(self,
                 inp_dim: int, out_dim: int, 
                 kinematic_emb: int, image_emb: int,
                 hidden_dims: int, fusion_dims: List[int], 
                 mobone_type: str, mobone_path: str,
                 fusion: str,
                 device: str):
        super(KIFNet, self).__init__()

        if fusion in ('concat', 'concat-ln', 'concat-bn'):
            decoder_output = image_emb + kinematic_emb
        else:
            decoder_output = image_emb
        
        self.kinematic_model = Single_Single(inp_dim=inp_dim,
                                             out_dim=kinematic_emb,
                                             hidden_dims=hidden_dims
                                            ).to(device)

        self.cv_model = VisionModel(embedding_dim=image_emb,
                                    mobone_type=mobone_type,
                                    mobone_path=mobone_path
                                   ).to(device)
        
        self.decoders = [
            Single_Single(inp_dim=decoder_output,
                          out_dim=1,
                          hidden_dims=fusion_dims
                         ).to(device)
            for _ in range(out_dim)]
        
        self.fusion_net = FusionNet(fusion, fusion_dims, decoder_output)

    def forward(self, X_kin, X_cv):
        kin = self.kinematic_model(X_kin)
        cv = self.cv_model(X_cv)
        
        fusion_out = self.fusion_net(kin, cv)
        outs = torch.cat([dec(fusion_out) for dec in self.decoders], dim=1)
        
        return outs
