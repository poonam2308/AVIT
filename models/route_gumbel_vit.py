import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import math

def reduce_scores(S: torch.Tensor, reduce: str, dim: int) -> torch.Tensor:
    if reduce == "max":
        return S.max(dim=dim).values
    if reduce == "mean":
        return S.mean(dim=dim)
    if reduce == "logsumexp":
        return torch.logsumexp(S, dim=dim)
    raise ValueError(f"Unknown reduce='{reduce}' (use max/mean/logsumexp)")


def sample_gumbel(shape, device, eps: float = 1e-6):
    # g = -log(-log(u))
    u = torch.rand(shape, device=device).clamp_(eps, 1 - eps)
    return -torch.log(-torch.log(u))

def diversity_usage_uniform(soft_w: torch.Tensor, eps: float = 1e-8):
    """
    soft_w: (B, N)
    Returns: scalar loss (minimize)
    Idea: maximize entropy of mean usage across batch.
    """
    p = soft_w.mean(dim=0)                 # (N,)
    p = p / (p.sum() + eps)
    entropy = -(p * (p + eps).log()).sum()
    return -entropy

def diversity_batch_cosine(soft_w: torch.Tensor, eps: float = 1e-8):
    """
    soft_w: (B, N)
    Penalize cosine similarity between samples.
    """
    w = soft_w / (soft_w.norm(dim=1, keepdim=True) + eps)  # (B,N)
    sim = w @ w.T                                          # (B,B)
    off_diag = sim - torch.eye(sim.size(0), device=sim.device)
    return off_diag.mean()

class RoutingSchedule:
    """
    Example schedule:
      - tau decays from tau0 -> tau_min
      - gumbel used early, turned off after `gumbel_off_step`
    """
    def __init__(self, tau0=2.0, tau_min=0.5, decay=0.9995, gumbel_off_step=20000):
        self.tau0 = tau0
        self.tau_min = tau_min
        self.decay = decay
        self.gumbel_off_step = gumbel_off_step

    def __call__(self, step: int):
        tau = max(self.tau_min, self.tau0 * (self.decay ** step))
        use_gumbel = step < self.gumbel_off_step
        return tau, use_gumbel

class STGumbelTopKRouteBlock(nn.Module):
    """
    Straight-through Gumbel-TopK selection:
      - Q from H0, K from X0 -> S = QK^T / sqrt(d)
      - collapse S -> scores (B, N)
      - training: add Gumbel noise, hard TopK indices for forward,
                  but use softmax(scores_noisy/tau) for backward
      - eval: hard TopK on scores (no noise)

    mode:
      - "tokens": score tokens i (reduce over patches), gather from H0
      - "patches": score patches j (reduce over tokens), gather from X0 or H0 (gather_from)
    """
    def __init__(
        self,
        embed_dim: int,
        qk_dim: int = 128,
        keep_k: int = 32,
        mode: str = "tokens",          # "tokens" | "patches"
        reduce: str = "logsumexp",     # "max" | "mean" | "logsumexp"
        gather_from: str = "h0",       # for mode="patches": "x0" or "h0"
        tau: float = 1.0,              # temperature for soft relaxation
    ):
        super().__init__()
        assert mode in ("tokens", "patches")
        assert gather_from in ("x0", "h0")
        self.mode = mode
        self.reduce = reduce
        self.gather_from = gather_from
        self.keep_k = keep_k
        self.tau = tau
        self.use_gumbel = True


        self.Wq = nn.Linear(embed_dim, qk_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, qk_dim, bias=False)
        self.scale = qk_dim ** -0.5

    def set_tau(self, tau: float):
        self.tau = float(tau)

    def set_use_gumbel(self, flag: bool):
        self.use_gumbel = bool(flag)

    def forward(self, X0_patches: torch.Tensor, H0_patches: torch.Tensor):
        """
        X0_patches: (B, N, D)
        H0_patches: (B, N, D)
        Returns:
          out_tokens: (B, K, D)  (K fixed in train and eval)
          debug: dict
        """
        B, N, D = H0_patches.shape
        Kkeep = min(self.keep_k, N)

        Q = self.Wq(H0_patches)  # (B, N, d)
        K = self.Wk(X0_patches)  # (B, N, d)
        S = torch.matmul(Q, K.transpose(-1, -2)) * self.scale  # (B, N, N)

        if self.mode == "tokens":
            # scores per token i (collapse over patches j)
            scores = reduce_scores(S, self.reduce, dim=-1)  # (B, N)
            source = H0_patches
        else:
            # scores per patch j (collapse over tokens i)
            scores = reduce_scores(S, self.reduce, dim=-2)  # (B, N)
            source = X0_patches if self.gather_from == "x0" else H0_patches

        if self.training:
            if self.use_gumbel:
                g = sample_gumbel(scores.shape, device=scores.device)
                scores_noisy = scores + g
            else:
                scores_noisy = scores
            idx = torch.topk(scores_noisy, k=Kkeep, dim=-1).indices  # (B, K)

            # soft relaxation over ALL N items
            soft_w = F.softmax(scores_noisy / self.tau, dim=-1)       # (B, N)

            # hard gathered tokens (forward path)
            hard_sel = torch.gather(source, 1, idx.unsqueeze(-1).expand(-1, -1, D))  # (B, K, D)

            # soft-weighted gathered tokens (backward path)
            w_sel = torch.gather(soft_w, 1, idx)                       # (B, K)
            soft_sel = hard_sel * w_sel.unsqueeze(-1)                  # (B, K, D)

            # straight-through: forward == hard_sel, backward flows through soft_sel
            out = hard_sel.detach() + soft_sel - soft_sel.detach()

            return out, {"idx": idx, "scores": scores, "scores_noisy": scores_noisy, "soft_w": soft_w}

        # eval: hard topk (no noise)
        idx = torch.topk(scores, k=Kkeep, dim=-1).indices
        out = torch.gather(source, 1, idx.unsqueeze(-1).expand(-1, -1, D))
        return out, {"idx": idx, "scores": scores}


class TimmViTWithTopKRouting_STGumbel(nn.Module):
    """
    timm ViT wrapper + ST Gumbel-TopK routing
      Step 1: patch embed + cls + pos
      Step 2: early blocks -> H0
      Step 3-5: STGumbelTopKRouteBlock -> (B, K, D) always
      Step 6: remaining blocks on [CLS + selected]
      Step 7: norm + head
    """
    def __init__(
        self,
        timm_name: str = "vit_base_patch16_224",
        pretrained: bool = False,
        num_classes: int = 10,
        split_block: int = 4,
        qk_dim: int = 128,
        keep_k: int = 32,
        mode: str = "tokens",          # "tokens" | "patches"
        reduce: str = "logsumexp",
        gather_from: str = "h0",       # for mode="patches": "x0" or "h0"
        tau: float = 1.0,
        add_routed_pos: bool = True,
    ):
        super().__init__()
        self.vit = timm.create_model(timm_name, pretrained=pretrained)
        self.vit.reset_classifier(num_classes=num_classes)

        for attr in ("patch_embed", "cls_token", "pos_embed", "pos_drop", "blocks", "norm", "head"):
            assert hasattr(self.vit, attr), f"Expected vit.{attr} to exist."

        self.split_block = split_block
        D = self.vit.embed_dim

        self.router = STGumbelTopKRouteBlock(
            embed_dim=D,
            qk_dim=qk_dim,
            keep_k=keep_k,
            mode=mode,
            reduce=reduce,
            gather_from=gather_from,
            tau=tau,
        )

        self.add_routed_pos = add_routed_pos
        self.keep_k = keep_k
        if add_routed_pos:
            self.routed_pos = nn.Parameter(torch.zeros(1, keep_k, D))
            nn.init.trunc_normal_(self.routed_pos, std=0.02)
        else:
            self.routed_pos = None

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        B = x.shape[0]

        # Step 1: patch embed + cls + pos
        x = self.vit.patch_embed(x)                        # (B, N, D)
        cls = self.vit.cls_token.expand(B, -1, -1)         # (B, 1, D)
        x = torch.cat((cls, x), dim=1)                     # (B, 1+N, D)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)

        X0_patches = x[:, 1:, :]                           # (B, N, D)

        # Step 2: early blocks
        h = x
        for blk in self.vit.blocks[: self.split_block]:
            h = blk(h)
        cls_h0 = h[:, :1, :]                               # (B, 1, D)
        H0_patches = h[:, 1:, :]                           # (B, N, D)

        # Step 3-5: ST Gumbel TopK routing (always returns (B, K, D))
        selected, dbg = self.router(X0_patches, H0_patches)  # (B, K, D)

        # after: selected, dbg = self.router(...)
        if isinstance(dbg, dict):
            # store for visualization utilities
            self.last_selected_idx = dbg.get("idx", None)
            self.last_scores = dbg.get("scores", None)

        if self.add_routed_pos:
            # if N < keep_k, router returns smaller Kkeep; handle safely
            L = selected.shape[1]
            selected = selected + self.routed_pos[:, :L, :]

        # Step 6: remaining blocks on reduced sequence
        h2 = torch.cat([cls_h0, selected], dim=1)          # (B, 1+K, D)
        for blk in self.vit.blocks[self.split_block:]:
            h2 = blk(h2)

        # Step 7: norm + head (CLS)
        h2 = self.vit.norm(h2)
        out = self.vit.head(h2[:, 0])

        if return_debug:
            return out, dbg
        return out
#
# model = TimmViTWithTopKRouting_STGumbel(
#     timm_name="vit_base_patch16_224",
#     pretrained=False,
#     num_classes=10,
#     split_block=4,
#     keep_k=32,
#     mode="tokens",
#     tau=1.0,
# )
#
# model = TimmViTWithTopKRouting_STGumbel(
#     timm_name="vit_base_patch16_224",
#     pretrained=False,
#     num_classes=10,
#     split_block=4,
#     keep_k=32,
#     mode="patches",
#     gather_from="x0",   # or "h0"
#     tau=1.0,
# )