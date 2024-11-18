import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch, seq_len, d_model = x.size()
        
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(batch, seq_len, d_model)
        
        return self.proj(x)

class ConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=31, expansion_factor=2, dropout=0.1):
        super().__init__()
        
        inner_dim = d_model * expansion_factor
        padding = (kernel_size - 1) // 2
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, inner_dim * 2, 1)
        self.glu = nn.GLU(dim=1)
        self.depth_conv = nn.Conv1d(
            inner_dim, inner_dim, kernel_size,
            padding=padding, groups=inner_dim
        )
        self.batch_norm = nn.BatchNorm1d(inner_dim)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(inner_dim, d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        residual = x
        x = self.layer_norm(x)
        
        # Conv1D expects (batch, channel, seq_len)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.glu(x)
        x = self.depth_conv(x)
        x = self.batch_norm(x)
        x = self.act(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)
        
        return residual + self.dropout(x)

class ConformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, kernel_size=31, expansion_factor=2, dropout=0.1):
        super().__init__()
        
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.conv = ConvModule(d_model, kernel_size, expansion_factor, dropout)
        
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(self.layer_norm(x), mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return x

class ConformerEncoder(nn.Module):
    def __init__(self, num_layers=6, d_model=512, num_heads=8, kernel_size=31, dropout=0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, num_heads, kernel_size, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.layer_norm(x)