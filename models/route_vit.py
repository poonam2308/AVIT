import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def reduce_scores(S: torch.Tensor, reduce: str, dim: int) -> torch.Tensor:
    if reduce == "max":
        return S.max(dim=dim).values
    if reduce == "mean":
        return S.mean(dim=dim)
    if reduce == "logsumexp":
        return torch.logsumexp(S, dim=dim)
    raise ValueError(f"Unknown reduce='{reduce}' (use max/mean/logsumexp)")


class SoftHardRouteBlock(nn.Module):
    """
    Train: soft routing -> (B, M, D) routed tokens (differentiable)
    Eval:  hard Top-K gather -> (B, K, D) selected tokens (fast)

    Q from H0, K from X0:
      Q = H0 Wq
      K = X0 Wk
      S = QK^T / sqrt(d)

    Then collapse S into scores:
      mode='tokens'  -> per-token scores (reduce over patches)
      mode='patches' -> per-patch scores (reduce over tokens)
    """
    def __init__(
        self,
        embed_dim: int,
        n_patches: int,
        qk_dim: int = 128,
        keep_k: int = 32,              # hard K at eval
        routed_m: int = 32,            # soft M during training (often = keep_k)
        mode: str = "tokens",          # "tokens" | "patches"
        reduce: str = "logsumexp",
        gather_from: str = "h0",       # for mode="patches": "x0" or "h0"
        tau_train: float = 1.0,        # temperature for soft routing over scores
    ):
        super().__init__()
        assert mode in ("tokens", "patches")
        assert gather_from in ("x0", "h0")
        self.mode = mode
        self.reduce = reduce
        self.keep_k = keep_k
        self.routed_m = routed_m
        self.gather_from = gather_from
        self.tau_train = tau_train

        self.n_patches = n_patches

        self.Wq = nn.Linear(embed_dim, qk_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, qk_dim, bias=False)
        self.scale = qk_dim ** -0.5

        # Learned slot weights to produce M routed outputs during training
        # slot_logits: (M, N)
        self.slot_logits = nn.Parameter(torch.randn(routed_m, n_patches) * 0.02)

    def forward(self, X0_patches: torch.Tensor, H0_patches: torch.Tensor):
        """
        X0_patches: (B, N, D) Step-1 patches (after pos)
        H0_patches: (B, N, D) Step-2 patches (after early blocks)
        Returns:
          out_tokens: (B, M, D) in train OR (B, K, D) in eval
          debug: dict
        """
        B, N, D = H0_patches.shape
        assert N == self.n_patches, f"Expected N={self.n_patches}, got {N}"

        Q = self.Wq(H0_patches)  # (B, N, d)
        K = self.Wk(X0_patches)  # (B, N, d)
        S = torch.matmul(Q, K.transpose(-1, -2)) * self.scale  # (B, N, N)

        if self.mode == "tokens":
            # scores per token i: reduce over patches j (columns)
            scores = reduce_scores(S, self.reduce, dim=-1)  # (B, N)
            source = H0_patches
        else:
            # scores per patch j: reduce over tokens i (rows)
            scores = reduce_scores(S, self.reduce, dim=-2)  # (B, N)
            source = X0_patches if self.gather_from == "x0" else H0_patches

        if not self.training:
            # EVAL: hard Top-K gather
            Kkeep = min(self.keep_k, N)
            idx = torch.topk(scores, k=Kkeep, dim=-1).indices  # (B, K)
            out = torch.gather(source, 1, idx.unsqueeze(-1).expand(-1, -1, D))  # (B, K, D)
            return out, {"idx": idx, "scores": scores}

        # TRAIN: soft routing to M outputs
        # per-example keep distribution over N items
        w_keep = F.softmax(scores / self.tau_train, dim=-1)  # (B, N)

        # per-slot prior distribution over N items
        slot_w = F.softmax(self.slot_logits, dim=-1)  # (M, N)

        # combine -> effective weights (B, M, N)
        w = slot_w.unsqueeze(0) * w_keep.unsqueeze(1)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)

        out = torch.matmul(w, source)  # (B, M, D)
        return out, {"weights": w, "scores": scores}


class TimmViTWithTopKRouting(nn.Module):
    """
    timm ViT wrapper with:
      - early blocks on full tokens
      - soft routing during training (M tokens)
      - hard Top-K during eval (K tokens)
      - remaining blocks on reduced tokens
    """
    def __init__(
        self,
        timm_name: str = "vit_base_patch16_224",
        pretrained: bool = False,
        num_classes: int = 10,
        split_block: int = 4,
        qk_dim: int = 128,
        keep_k: int = 32,
        routed_m: int = 32,
        mode: str = "tokens",          # "tokens" | "patches"
        reduce: str = "logsumexp",     # "max" | "mean" | "logsumexp"
        gather_from: str = "h0",       # for mode="patches": "x0" or "h0"
        tau_train: float = 1.0,
        add_routed_pos: bool = True,   # optional: positional for routed tokens
    ):
        super().__init__()
        self.vit = timm.create_model(timm_name, pretrained=pretrained)
        self.vit.reset_classifier(num_classes=num_classes)

        # Required timm VisionTransformer parts
        for attr in ("patch_embed", "cls_token", "pos_embed", "pos_drop", "blocks", "norm", "head"):
            assert hasattr(self.vit, attr), f"Expected vit.{attr} to exist."

        self.split_block = split_block
        D = self.vit.embed_dim

        # infer patch count from pos_embed (1 + N)
        n_tokens = self.vit.pos_embed.shape[1]
        self.n_patches = n_tokens - 1

        self.router = SoftHardRouteBlock(
            embed_dim=D,
            n_patches=self.n_patches,
            qk_dim=qk_dim,
            keep_k=keep_k,
            routed_m=routed_m,
            mode=mode,
            reduce=reduce,
            gather_from=gather_from,
            tau_train=tau_train,
        )

        self.add_routed_pos = add_routed_pos
        self.keep_k = keep_k
        self.routed_m = routed_m

        if add_routed_pos:
            # We need pos for whichever length we will see:
            # train: (M), eval: (K). Easiest: allocate max(M,K) and slice.
            max_len = max(keep_k, routed_m)
            self.routed_pos = nn.Parameter(torch.zeros(1, max_len, D))
            nn.init.trunc_normal_(self.routed_pos, std=0.02)
        else:
            self.routed_pos = None

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        B = x.shape[0]

        # ---- Step 1: patch embed + cls + pos
        x = self.vit.patch_embed(x)                         # (B, N, D)
        cls = self.vit.cls_token.expand(B, -1, -1)          # (B, 1, D)
        x = torch.cat((cls, x), dim=1)                      # (B, 1+N, D)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)

        X0_patches = x[:, 1:, :]                            # (B, N, D)

        # ---- Step 2: early blocks on full sequence
        h = x
        for blk in self.vit.blocks[: self.split_block]:
            h = blk(h)

        cls_h0 = h[:, :1, :]                                # (B, 1, D)
        H0_patches = h[:, 1:, :]                            # (B, N, D)

        # ---- Step 3-5: routing (train: soft M; eval: hard K)
        routed_or_selected, dbg = self.router(X0_patches, H0_patches)
        # routed_or_selected: (B, M, D) in train OR (B, K, D) in eval

        # Optional positional for routed tokens
        if self.add_routed_pos:
            L = routed_or_selected.shape[1]
            routed_or_selected = routed_or_selected + self.routed_pos[:, :L, :]

        # ---- Step 6: remaining blocks on reduced sequence
        h2 = torch.cat([cls_h0, routed_or_selected], dim=1)  # (B, 1+L, D)
        for blk in self.vit.blocks[self.split_block:]:
            h2 = blk(h2)

        # ---- Step 7: norm + head on CLS
        h2 = self.vit.norm(h2)
        out = self.vit.head(h2[:, 0])

        if return_debug:
            return out, dbg
        return out