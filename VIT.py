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
from MultiHeadAttention import MultiHeadAttention
from EncoderVIT import EncoderVIT
torch.manual_seed(0)

class VIT(nn.Module):
    def __init__(self, chw=(1,28,28), n_patches=7, n_heads=2, n_blocks=2, hidden_d=2, out_d=10):
        super(VIT, self).__init__()

        self.chw = chw 
        self.n_patches = n_patches
        self.hidden_d = hidden_d
        self.n_blocks = n_blocks
        self.n_heads = n_heads 
        self.out_d = out_d

        assert chw[1] % n_patches == 0, 'Input shape should be divisible by n_patches'
        assert chw[2] % n_patches == 0, 'Input shape should be divisible by n_patches'

        self.patch_size = (chw[1] // n_patches, chw[2] // n_patches)

        self.input_d = int(self.chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        self.class_token = nn.Parameter(torch.randn(1, self.hidden_d))

        self.pos_embed = nn.Parameter(torch.randn(self.n_patches**2 + 1, self.hidden_d))
        self.pos_embed.requires_grad = False 

        self.encoder_blocks = nn.ModuleList([EncoderVIT(self.hidden_d, n_heads) for _ in range(n_blocks)])

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, images):
        n, c, h, w = images.shape
        patches = self.patch_embedding(images, self.n_patches)
        tokens = self.linear_mapper(patches.to(images.device))
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

        pos_embed = self.pos_embed.repeat(n, 1, 1).to(images.device)
        out = tokens + pos_embed

        for block in self.encoder_blocks:
            out = block(out)

        out = out[:, 0]
        
        return self.mlp(out)
    
    @staticmethod
    def patch_embedding(images, n_patches):
        n, c, h, w = images.shape

        assert h == w, 'Patch embedding requires the dimensions of the height and width to be the same'

        patches = torch.zeros(n, n_patches**2, h * w * c // n_patches**2, device=images.device)
        patch_size = h // n_patches

        for idx, image in enumerate(images):
            for i in range(n_patches):
                for j in range(n_patches):
                    patch = image[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                    patches[idx, i*n_patches + j] = patch.flatten()

        return patches

    @staticmethod
    def positional_embedding(sequence_length, d):
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return result