import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast as autocast
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        if num_heads is None:
            num_heads = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = x.transpose(1, 2)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(1, 2)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_layers, dim, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([
            Attention(dim, num_heads=num_heads) for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Example usage
if __name__ == "__main__":
  num_layers = 3
  dim = 2049
  num_heads = 8

  multi_layer_attention = MultiHeadAttention(num_layers,dim, num_heads=num_heads).cuda()

   