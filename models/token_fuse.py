import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class TokenFusion(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gamma = nn.Parameter(torch.zeros(1)) # Start at 0
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, patch_toks, selected_tokens):
        out, _ = self.attn(
            query=patch_toks,
            key=selected_tokens,
            value=selected_tokens,
        )
        x = self.norm1(patch_toks + out)
        x = self.norm2(x + self.mlp(x))
        return self.gamma * x