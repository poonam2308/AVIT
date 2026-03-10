import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


import math
import torch
import torch.nn as nn


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: [M]
    returns: [M, embed_dim]
    """
    assert embed_dim % 2 == 0

    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2.0)))

    pos = pos.reshape(-1)  # [M]
    out = torch.einsum("m,d->md", pos, omega)  # [M, D/2]

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    return torch.cat([emb_sin, emb_cos], dim=1)  # [M, D]


def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w, device):
    """
    returns: [grid_h * grid_w, embed_dim]
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sin-cos embedding"

    grid_y = torch.arange(grid_h, dtype=torch.float32, device=device)
    grid_x = torch.arange(grid_w, dtype=torch.float32, device=device)

    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")  # [H, W], [H, W]

    yy = yy.reshape(-1)  # [H*W]
    xx = xx.reshape(-1)  # [H*W]

    emb_y = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, yy)  # [H*W, D/2]
    emb_x = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, xx)  # [H*W, D/2]

    return torch.cat([emb_y, emb_x], dim=1)  # [H*W, D]


class OverlapPatchEmbed_old(nn.Module):
    def __init__(self, in_chans=3, patch_size=16, stride=8, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=0,
            bias=True,
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        feat = self.proj(x)                         # [B, C, H', W']
        B, C, Hs, Ws = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)   # [B, N_new, C]
        return tokens, (Hs, Ws)

    def forward_with_pos(self, x):
        tokens, (Hs, Ws) = self.forward(x)
        pos = get_2d_sincos_pos_embed(
            embed_dim=tokens.shape[-1],
            grid_h=Hs,
            grid_w=Ws,
            device=tokens.device,
        )  # [N_new, C]
        tokens = tokens + pos.unsqueeze(0)         # [B, N_new, C]
        return tokens, (Hs, Ws), pos


# class OverlapPatchEmbed(nn.Module):
#     def __init__(self, in_chans=3, patch_size=16, stride=8, embed_dim=768):
#         super().__init__()
#         self.patch_size = patch_size
#         self.stride = stride
#         self.proj = nn.Conv2d(
#             in_chans,
#             embed_dim,
#             kernel_size=patch_size,
#             stride=stride,
#             padding=0,
#             bias=True,
#         )
#
#     def forward(self, x):
#         # x: [B, 3, H, W]
#         feat = self.proj(x)                    # [B, C, H', W']
#         B, C, Hs, Ws = feat.shape
#         tokens = feat.flatten(2).transpose(1, 2)   # [B, N_new, C]
#         return tokens, (Hs, Ws)



class Learned2DPosEmbed(nn.Module):
    def __init__(self, embed_dim, base_grid_size=27):
        super().__init__()
        self.embed_dim = embed_dim
        self.base_grid_size = base_grid_size
        self.pos = nn.Parameter(
            torch.zeros(1, embed_dim, base_grid_size, base_grid_size)
        )
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, H, W):
        pos = F.interpolate(
            self.pos,
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        )  # [1,C,H,W]
        pos = pos.flatten(2).transpose(1, 2)   # [1,H*W,C]
        return pos


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_chans=3, patch_size=16, stride=8, embed_dim=768, base_grid_size=27):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=0,
            bias=True,
        )
        self.pos_embed = Learned2DPosEmbed(embed_dim, base_grid_size=base_grid_size)

    def forward(self, x):
        feat = self.proj(x)                         # [B,C,Hs,Ws]
        B, C, Hs, Ws = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)   # [B,Ns,C]
        return tokens, (Hs, Ws)

    def forward_with_pos(self, x):
        tokens, (Hs, Ws) = self.forward(x)
        pos = self.pos_embed(Hs, Ws)               # [1,Ns,C]
        tokens = tokens + pos
        return tokens, (Hs, Ws), pos