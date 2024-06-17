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
from EncoderVIT import EncoderVIT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

class MultiHeadAttention(nn.Module):
    def __init__(self, d, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.d = d 
        self.n_heads = n_heads

        assert d % n_heads == 0, f'Dimension {d} is not divisible by head: {n_heads}'

        d_head = int(d / n_heads)
        self.q = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head 
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        for sequence in sequences:
            seq_res = []
            for head in range(self.n_heads):
                q_mapping = self.q[head]
                k_mapping = self.k[head]
                v_mapping = self.v[head]
                seq = sequence[:, head * self.d_head : (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_res.append(attention @ v)
            result.append(torch.hstack(seq_res))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

