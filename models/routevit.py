import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import math



def build_full_token_mask(patch_mask: torch.Tensor) -> torch.Tensor:
    """
    patch_mask: (B, N) with 1 for kept tokens, 0 for masked tokens
    returns:    (B, 1+N) with CLS always kept
    """
    B = patch_mask.shape[0]
    cls_mask = torch.ones(B, 1, device=patch_mask.device, dtype=patch_mask.dtype)
    return torch.cat([cls_mask, patch_mask], dim=1)


def masked_attention_forward(attn, x, full_mask):
    """
    Run timm Attention with additive masking on key positions.

    x:         (B, L, C)
    full_mask: (B, L), 1=keep, 0=mask
    """
    B, L, C = x.shape
    qkv = attn.qkv(x).reshape(B, L, 3, attn.num_heads, C // attn.num_heads)
    qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, Dh)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn_scores = (q @ k.transpose(-2, -1)) * attn.scale   # (B, H, L, L)

    # mask keys: masked tokens cannot be attended to
    key_keep = full_mask[:, None, None, :]                 # (B,1,1,L)
    attn_scores = attn_scores.masked_fill(key_keep < 0.5, -1e4)

    attn_probs = attn_scores.softmax(dim=-1)
    attn_probs = attn.attn_drop(attn_probs)

    out = (attn_probs @ v).transpose(1, 2).reshape(B, L, C)
    out = attn.proj(out)
    out = attn.proj_drop(out)
    return out


def masked_block_forward_(blk, x, full_mask):
    """
    Re-implement a timm ViT block forward using masked attention.
    Assumes DeiT/timm-style block with norm1, attn, drop_path, norm2, mlp.
    """
    x = x + blk.drop_path(masked_attention_forward(blk.attn, blk.norm1(x), full_mask))
    x = x + blk.drop_path(blk.mlp(blk.norm2(x)))
    return x
def masked_block_forward(blk, x, full_mask):
    """
    Re-implement a timm ViT block forward using masked attention.
    Supports both:
      - blk.drop_path
      - blk.drop_path1 / blk.drop_path2
      - no drop path attributes
    """
    attn_out = masked_attention_forward(blk.attn, blk.norm1(x), full_mask)
    mlp_out = blk.mlp(blk.norm2(x))

    if hasattr(blk, "drop_path1") and hasattr(blk, "drop_path2"):
        x = x + blk.drop_path1(attn_out)
        x = x + blk.drop_path2(mlp_out)
    elif hasattr(blk, "drop_path"):
        x = x + blk.drop_path(attn_out)
        x = x + blk.drop_path(mlp_out)
    else:
        x = x + attn_out
        x = x + mlp_out

    return x

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
      - Q from X0, K from H0 -> S = QK^T / sqrt(d)
      - reduce over contextual keys to get one score per input token in X0
      - training: add Gumbel noise, hard TopK indices for forward,
                  but use softmax(scores_noisy/tau) for backward
      - eval: hard TopK on scores (no noise)

    Intuition:
      - X0 tokens are the candidate input tokens we want to rank/select
      - H0 tokens provide contextual information used to judge usefulness
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

        # Your intended flow:
        #   queries come from input tokens X0
        #   keys come from contextual tokens H0
        Q = self.Wq(X0_patches)  # (B, N, d)
        K = self.Wk(H0_patches)  # (B, N, d)
        S = torch.matmul(Q, K.transpose(-1, -2)) * self.scale  # (B, N_x0, N_h0)

        # We want one score per input token in X0, so reduce over contextual keys.
        scores = reduce_scores(S, self.reduce, dim=-1)  # (B, N)

        # Hard-gathered tokens for ST path should come from X0 because we are selecting X0 positions.
        source = X0_patches

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

class RouteGumbelViTTokenReduction(nn.Module):
    """
    Normal ViT-style classifier with routing-aware token reduction.

    Output:
      - forward(x) -> logits
      - forward(x, return_debug=True) -> logits, dbg

    Flow:
      1) Run ViT normally up to split_block
      2) X0 = patch tokens before early contextualization
      3) H0 = patch tokens after early contextualization
      4) Router scores X0 tokens using H0 context
      5) Top-k selected X0 positions are gathered
      6) Selected X0 information is fused into selected H0 tokens
      7) Remaining blocks run only on [CLS] + K selected tokens
      8) CLS token produces logits
    """

    def __init__(
        self,
        timm_name="vit_base_patch16_224",
        pretrained=False,
        num_classes=10,
        split_block=4,
        qk_dim=128,
        keep_k=32,
        mode="tokens",
        reduce="logsumexp",
        gather_from="x0",
        tau=1.0,
        add_routed_pos=True,
    ):
        super().__init__()

        self.vit = timm.create_model(timm_name, pretrained=pretrained)
        self.vit.reset_classifier(num_classes=num_classes)

        for attr in ("patch_embed", "cls_token", "pos_embed", "pos_drop",
                     "blocks", "norm", "head"):
            assert hasattr(self.vit, attr), f"Expected vit.{attr} to exist."

        self.num_classes = num_classes
        self.num_features = self.embed_dim = self.vit.embed_dim
        self.split_block = split_block
        self.add_routed_pos = add_routed_pos
        self.keep_k = keep_k

        self.router = STGumbelTopKRouteBlock(
            embed_dim=self.embed_dim,
            qk_dim=qk_dim,
            keep_k=keep_k,
            mode=mode,
            reduce=reduce,
            gather_from=gather_from,
            tau=tau,
        )

        self.routed_pos = nn.Parameter(torch.zeros(1, keep_k, self.embed_dim))
        nn.init.trunc_normal_(self.routed_pos, std=0.02)

        self.x0_to_h = nn.Linear(self.embed_dim, self.embed_dim)
        self.fuse_alpha = nn.Parameter(torch.tensor(1.0))

        self.last_selected_idx = None
        self.last_scores = None

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"vit.pos_embed", "vit.cls_token", "routed_pos"}

    def get_classifier(self):
        return self.vit.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.vit.reset_classifier(num_classes)

    def set_tau(self, tau: float):
        if hasattr(self.router, "set_tau"):
            self.router.set_tau(tau)
        else:
            self.router.tau = tau

    def set_use_gumbel(self, flag: bool):
        if hasattr(self.router, "set_use_gumbel"):
            self.router.set_use_gumbel(flag)
        else:
            self.router.use_gumbel = flag

    def forward(self, x, return_debug: bool = False):
        B = x.shape[0]

        # 1) Patch embed + CLS + pos
        x = self.vit.patch_embed(x)                 # (B, N, D)
        N = x.shape[1]

        cls = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)             # (B, 1+N, D)
        x = x + self.vit.pos_embed[:, :N + 1, :]
        x = self.vit.pos_drop(x)

        # X0 = input patch tokens before early contextualization
        X0_patches = x[:, 1:, :]                   # (B, N, D)

        # 2) Early blocks
        h = x
        for blk in self.vit.blocks[:self.split_block]:
            h = blk(h)

        cls_h0 = h[:, :1, :]                       # (B, 1, D)
        H0_patches = h[:, 1:, :]                   # (B, N, D)

        # 3) Router selects useful X0 positions using H0 context
        selected_x0, dbg = self.router(X0_patches, H0_patches)
        idx = dbg["idx"]                           # (B, K)
        scores = dbg.get("scores", None)
        soft_w = dbg.get("soft_w", None)

        self.last_selected_idx = idx
        self.last_scores = scores

        Ksel = idx.shape[1]

        # Gather selected H0 tokens at same chosen positions
        selected_h0 = torch.gather(
            H0_patches,
            dim=1,
            index=idx.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        )                                          # (B, K, D)

        # Optional routed positional signal
        if self.add_routed_pos:
            selected_x0 = selected_x0 + self.routed_pos[:, :Ksel, :]

        # 4) Fuse selected X0 into selected H0
        selected_fused = selected_h0 + self.fuse_alpha * self.x0_to_h(selected_x0)

        # 5) Reduced sequence for later blocks
        h2 = torch.cat([cls_h0, selected_fused], dim=1)   # (B, 1+K, D)

        for blk in self.vit.blocks[self.split_block:]:
            h2 = blk(h2)

        h2 = self.vit.norm(h2)

        # 6) Normal classifier output
        logits = self.vit.head(h2[:, 0])

        if return_debug:
            patch_mask = H0_patches.new_zeros(B, N)
            patch_mask.scatter_(1, idx, 1.0)

            dbg_out = {
                "logits": logits,
                "x0": X0_patches,
                "h0": H0_patches,
                "selected_x0": selected_x0,
                "selected_h0": selected_h0,
                "selected_fused": selected_fused,
                "selected_idx": idx,
                "scores": scores,
                "soft_w": soft_w,
                "patch_mask": patch_mask,
                "compact_tokens": h2[:, 1:, :],
                "cls_h0": cls_h0,
            }
            return logits, dbg_out

        return logits

    def forward_debug(self, x):
        self.eval()
        with torch.no_grad():
            _, dbg = self.forward(x, return_debug=True)
        return dbg

class RouteGumbelViTTokenEmphasis(nn.Module):
    """
    Normal ViT-style classifier with routing-aware token emphasis.

    Output:
      - forward(x) -> logits
      - forward(x, return_debug=True) -> logits, dbg

    Flow:
      1) Run ViT normally up to split_block
      2) X0 = patch tokens before early contextualization
      3) H0 = patch tokens after early contextualization
      4) Router scores X0 tokens using H0 context
      5) Top-k defines selected positions
      6) Build a full patch mask over original N positions
      7) Reweight / fuse tokens, but KEEP full sequence length
      8) Later blocks use masked attention
      9) CLS token produces logits
    """

    def __init__(
        self,
        timm_name="vit_base_patch16_224",
        pretrained=False,
        num_classes=10,
        split_block=4,
        qk_dim=128,
        keep_k=32,
        mode="tokens",
        reduce="logsumexp",
        gather_from="x0",
        tau=1.0,
        add_routed_pos=True,
    ):
        super().__init__()

        self.vit = timm.create_model(timm_name, pretrained=pretrained)
        self.vit.reset_classifier(num_classes=num_classes)

        for attr in (
            "patch_embed", "cls_token", "pos_embed", "pos_drop",
            "blocks", "norm", "head"
        ):
            assert hasattr(self.vit, attr), f"Expected vit.{attr} to exist."

        self.num_classes = num_classes
        self.num_features = self.embed_dim = self.vit.embed_dim
        self.split_block = split_block
        self.add_routed_pos = add_routed_pos
        self.keep_k = keep_k

        self.router = STGumbelTopKRouteBlock(
            embed_dim=self.embed_dim,
            qk_dim=qk_dim,
            keep_k=keep_k,
            mode=mode,
            reduce=reduce,
            gather_from=gather_from,
            tau=tau,
        )

        self.routed_pos = nn.Parameter(torch.zeros(1, keep_k, self.embed_dim))
        nn.init.trunc_normal_(self.routed_pos, std=0.02)

        self.x0_to_h = nn.Linear(self.embed_dim, self.embed_dim)
        self.fuse_alpha = nn.Parameter(torch.tensor(1.0))

        self.last_selected_idx = None
        self.last_scores = None

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"vit.pos_embed", "vit.cls_token", "routed_pos"}

    def get_classifier(self):
        return self.vit.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.vit.reset_classifier(num_classes)

    def set_tau(self, tau: float):
        if hasattr(self.router, "set_tau"):
            self.router.set_tau(tau)
        else:
            self.router.tau = tau

    def set_use_gumbel(self, flag: bool):
        if hasattr(self.router, "set_use_gumbel"):
            self.router.set_use_gumbel(flag)
        else:
            self.router.use_gumbel = flag

    def forward(self, x, return_debug: bool = False):
        B = x.shape[0]

        # 1) Patch embed + CLS + pos
        x = self.vit.patch_embed(x)                 # (B, N, D)
        N = x.shape[1]

        cls = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)             # (B, 1+N, D)
        x = x + self.vit.pos_embed[:, :N + 1, :]
        x = self.vit.pos_drop(x)

        X0_patches = x[:, 1:, :]                   # (B, N, D)

        # 2) Early blocks
        h = x
        for blk in self.vit.blocks[:self.split_block]:
            h = blk(h)

        cls_h0 = h[:, :1, :]
        H0_patches = h[:, 1:, :]                   # (B, N, D)

        # 3) Router scores useful X0 positions using H0 context
        selected_x0, dbg = self.router(X0_patches, H0_patches)

        idx = dbg["idx"]
        soft_w = dbg.get("soft_w", None)
        scores = dbg.get("scores", None)

        self.last_selected_idx = idx
        self.last_scores = scores

        Ksel = idx.shape[1]

        # 4) Build hard mask over original token positions
        patch_mask = H0_patches.new_zeros(B, N)
        patch_mask.scatter_(1, idx, 1.0)

        # 5) Gate / emphasize X0 token content
        if self.training and soft_w is not None:
            x0_gated = X0_patches * soft_w.unsqueeze(-1)
        else:
            x0_gated = X0_patches * patch_mask.unsqueeze(-1)

        # Optional routed positional signal only at selected positions
        routed_pos_full = None
        if self.add_routed_pos:
            routed_pos_full = X0_patches.new_zeros(B, N, self.embed_dim)
            routed_pos_full.scatter_(
                1,
                idx.unsqueeze(-1).expand(-1, -1, self.embed_dim),
                self.routed_pos[:, :Ksel, :].expand(B, -1, -1)
            )
            x0_gated = x0_gated + routed_pos_full

        # 6) Fuse gated X0 into contextual H0
        fused_patches = H0_patches + self.fuse_alpha * self.x0_to_h(x0_gated)

        # 7) Keep full sequence length, use masked attention later
        h2 = torch.cat([cls_h0, fused_patches], dim=1)    # (B, 1+N, D)
        full_mask = build_full_token_mask(patch_mask)

        for blk in self.vit.blocks[self.split_block:]:
            h2 = masked_block_forward(blk, h2, full_mask)

        h2 = self.vit.norm(h2)

        # 8) Normal classifier output
        logits = self.vit.head(h2[:, 0])

        if return_debug:
            dbg_out = {
                "logits": logits,
                "x0": X0_patches,
                "h0": H0_patches,
                "selected_x0": selected_x0,
                "selected_idx": idx,
                "scores": scores,
                "soft_w": soft_w,
                "patch_mask": patch_mask,
                "full_mask": full_mask,
                "x0_gated": x0_gated,
                "fused_patches": fused_patches,
                "token_features": h2[:, 1:, :],
                "cls_h0": cls_h0,
                "routed_pos_full": routed_pos_full,
            }
            return logits, dbg_out

        return logits

    def forward_debug(self, x):
        self.eval()
        with torch.no_grad():
            _, dbg = self.forward(x, return_debug=True)
        return dbg