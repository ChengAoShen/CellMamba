import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MambaBlock(nn.Module):
    def __init__(self,in_channels,out_channels=None,last=False):
        super(MambaBlock,self).__init__()
        ...
