# gumbel_masked_vit.py
# A complete, ready-to-train PyTorch implementation of the "Option 1: Masking" sketch:
# Image -> PatchEmbed -> (Warm-up SA blocks) -> (Dynamic Gumbel-masked SA+MLP blocks) -> Head
#
# Key idea: per block we sample a per-token gate m (via Gumbel-Sigmoid) and use it to
#           mask attention logits for keys/values.


# gumbel_masked_vit.py
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Gumbel gate + regularizers
# -------------------------
def gumbel_sigmoid(
    logits: torch.Tensor,
    tau: float,
    training: bool,
    hard: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    logits: [B, N]
    returns: [B, N] in (0,1) (soft) or {0,1} with straight-through (hard=True)
    """
    if training:
        u = torch.rand_like(logits, dtype=torch.float32).clamp(eps, 1 - eps)
        g = torch.log(u) - torch.log(1 - u)  # logistic noise
        y = torch.sigmoid((logits + g) / tau)
        y = y.to(logits.dtype)
    else:
        y = torch.sigmoid(logits)

    if hard:
        y_hard = (y > 0.5).to(y.dtype)
        y = y_hard + (y - y.detach())  # straight-through estimator
    return y


def budget_loss(gates: List[torch.Tensor], target_keep_ratio: float) -> torch.Tensor:
    """
    gates: list of [B, N] (soft or ST)
    """
    if len(gates) == 0:
        return torch.tensor(0.0, device=gates[0].device if gates else "cpu")
    losses = []
    for m in gates:
        keep = m.mean(dim=1)  # [B]
        losses.append(((keep - target_keep_ratio) ** 2).mean())
    return torch.stack(losses).mean()


def entropy_loss(gates: List[torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    """
    gates: list of [B, N]
    """
    if len(gates) == 0:
        return torch.tensor(0.0, device=gates[0].device if gates else "cpu")
    losses = []
    for m in gates:
        m = torch.nan_to_num(m, nan=0.5, posinf=1.0, neginf=0.0)
        m = m.clamp(eps, 1 - eps)
        ent = -(m * torch.log(m) + (1 - m) * torch.log(1 - m))
        losses.append(ent.mean())
    return torch.stack(losses).mean()


# -------------------------
# Core building blocks
# -------------------------
class PatchEmbed(nn.Module):
    """Conv-based patch embedding (ViT style)."""
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                 # [B, D, Gh, Gw]
        x = x.flatten(2).transpose(1, 2) # [B, N, D]
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiHeadSelfAttentionMasked(nn.Module):
    """
    Manual MHA to support additive mask on attention logits.
    attn_additive_mask: [B, N_keys] added to logits for each key token (broadcasted).
    """
    def __init__(self, dim: int, num_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor, attn_additive_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x)  # [B, N, 3D]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, Hd]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        if attn_additive_mask is not None:
            attn = attn + attn_additive_mask[:, None, None, :]  # [B,1,1,N_keys]

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block (no gating)."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float, attn_dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttentionMasked(dim, num_heads, attn_dropout=attn_dropout, proj_dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_additive_mask=None)
        x = x + self.mlp(self.norm2(x))
        return x


class DynamicMaskedBlock(nn.Module):
    """
    Gate predictor -> Gumbel gates m -> mask attention logits for keys -> residual
    MLP update optionally gated by m -> residual.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
        gate_hidden: int = 128,
        gate_mlp: bool = True,
        gate_mlp_updates: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttentionMasked(dim, num_heads, attn_dropout=attn_dropout, proj_dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

        self.gate_mlp_updates = gate_mlp_updates

        if gate_mlp:
            self.gate_pred = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, 1),
            )
        else:
            self.gate_pred = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 1),
            )

    def forward(
        self,
        x: torch.Tensor,
        tau: float,
        training: bool,
        hard_gates: bool,
        inference_topk: Optional[int] = None,
        eps_mask: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        logits = self.gate_pred(x).squeeze(-1)  # [B, N]
        logits[:, 0] = 1e9  # keep CLS always

        if (not training) and (inference_topk is not None):
            m = torch.zeros_like(logits)
            m[:, 0] = 1.0
            k = min(inference_topk, N - 1)
            idx = torch.topk(logits[:, 1:], k=k, dim=1).indices + 1
            m.scatter_(dim=1, index=idx, value=1.0)
        else:
            m = gumbel_sigmoid(logits, tau=tau, training=training, hard=hard_gates)

        if torch.isnan(m).any() or torch.isinf(m).any():
            print("NaN/Inf detected in gates m")
            print("logits stats:", logits.min().item(), logits.max().item())
            print("m stats:", m.min().item(), m.max().item())
            raise RuntimeError("NaN/Inf in gates")
        # Attention key mask as additive logits: log(m+eps) <= 0
        attn_add_mask = torch.log(m.clamp(eps_mask, 1.0))  # [B, N_keys]

        # SA
        x = x + self.attn(self.norm1(x), attn_additive_mask=attn_add_mask)

        # MLP
        v = self.mlp(self.norm2(x))
        if self.gate_mlp_updates:
            v = v * m[:, :, None]
        x = x + v

        return x, m


# -------------------------
# Full model
# -------------------------
@dataclass
class MaskedViTConfig:
    img_size: int = 32
    patch_size: int = 4
    in_chans: int = 3
    num_classes: int = 10

    embed_dim: int = 256
    depth: int = 8
    warmup_depth: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0

    dropout: float = 0.1
    attn_dropout: float = 0.1

    gate_mlp: bool = True
    gate_hidden: int = 128
    gate_mlp_updates: bool = True

    # gating / losses
    target_keep_ratio: float = 0.6
    lambda_budget: float = 1.0
    lambda_entropy: float = 0.01

    # gumbel
    tau_start: float = 1.0
    tau_end: float = 0.2
    tau_anneal_steps: int = 20000
    hard_gates_train: bool = False

    # inference
    inference_topk: Optional[int] = None


class MaskedViT(nn.Module):
    def __init__(self, cfg: MaskedViTConfig):
        super().__init__()
        self.cfg = cfg

        self.patch = PatchEmbed(cfg.img_size, cfg.patch_size, cfg.in_chans, cfg.embed_dim)
        num_patches = self.patch.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.embed_dim))
        self.pos_drop = nn.Dropout(cfg.dropout)

        self.warmup_blocks = nn.ModuleList([
            TransformerBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout, cfg.attn_dropout)
            for _ in range(cfg.warmup_depth)
        ])

        dyn_depth = cfg.depth - cfg.warmup_depth
        assert dyn_depth >= 0
        self.dynamic_blocks = nn.ModuleList([
            DynamicMaskedBlock(
                cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout, cfg.attn_dropout,
                gate_hidden=cfg.gate_hidden, gate_mlp=cfg.gate_mlp, gate_mlp_updates=cfg.gate_mlp_updates
            )
            for _ in range(dyn_depth)
        ])

        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_tau(self, global_step: int) -> float:
        if self.cfg.tau_anneal_steps <= 0:
            return self.cfg.tau_end
        t = min(max(global_step, 0), self.cfg.tau_anneal_steps)
        frac = t / self.cfg.tau_anneal_steps
        return (1 - frac) * self.cfg.tau_start + frac * self.cfg.tau_end

    def forward(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        global_step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        B = images.shape[0]
        x = self.patch(images)  # [B, N, D]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.shape[1], :]
        x = self.pos_drop(x)

        # warm-up
        for blk in self.warmup_blocks:
            x = blk(x)

        tau = self.get_tau(global_step)
        gates: List[torch.Tensor] = []

        # dynamic
        training = self.training
        for blk in self.dynamic_blocks:
            x, m = blk(
                x,
                tau=tau,
                training=training,
                hard_gates=self.cfg.hard_gates_train,
                inference_topk=(self.cfg.inference_topk if (not training) else None),
            )
            gates.append(m)

        logits = self.head(self.norm(x[:, 0, :]))  # CLS
        out: Dict[str, torch.Tensor] = {"logits": logits, "tau": logits.new_tensor(tau)}

        if labels is not None:
            loss_task = F.cross_entropy(logits, labels)
            loss_b = budget_loss(gates, self.cfg.target_keep_ratio) if gates else logits.new_tensor(0.0)
            loss_e = entropy_loss(gates) if gates else logits.new_tensor(0.0)
            loss = loss_task + self.cfg.lambda_budget * loss_b + self.cfg.lambda_entropy * loss_e
            out.update({
                "loss": loss,
                "loss_task": loss_task.detach(),
                "loss_budget": loss_b.detach(),
                "loss_entropy": loss_e.detach(),
            })

        if gates:
            keep_ratio_mean = torch.stack([g.mean() for g in gates]).mean()
        else:
            keep_ratio_mean = logits.new_tensor(1.0)
        out["keep_ratio_mean"] = keep_ratio_mean.detach()

        return out