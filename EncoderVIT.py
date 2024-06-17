import numpy as np 
import torch 
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss 
from torch.optim import Adam 
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor 
from torchvision.datasets.mnist import MNIST 
from torchvision import transforms
from torch import nn
from VIT import VIT
from MultiHeadAttention import MultiHeadAttention
torch.manual_seed(0)




class EncoderVIT(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(EncoderVIT, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MultiHeadAttention(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out
        