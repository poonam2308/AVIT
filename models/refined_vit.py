import torch
import torch.nn as nn
import timm


class TopKTokenSelector(nn.Module):
    """
    Scores each token with an MLP and selects top-k patch tokens (keeps CLS always).
    """
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, k: int):
        """
        x: [B, N, D] (token 0 is CLS)
        returns:
          x_reduced: [B, 1+k, D]
          idx_full: [B, 1+k] token indices in the original x (including CLS index 0)
          scores: [B, N] (optional for debugging/visualization)
        """
        B, N, D = x.shape
        scores = self.scorer(x).squeeze(-1)          # [B, N]
        scores[:, 0] = 1e9                           # always keep CLS

        # select top-k among patch tokens only (exclude CLS from competition)
        k = min(k, N - 1)
        patch_scores = scores[:, 1:]                 # [B, N-1]
        topk_idx = patch_scores.topk(k, dim=1).indices + 1  # [B, k], shift back to token indices

        # build index list with CLS + selected
        cls_idx = torch.zeros(B, 1, device=x.device, dtype=topk_idx.dtype)
        idx_full = torch.cat([cls_idx, topk_idx], dim=1)     # [B, 1+k]

        # gather tokens
        idx_exp = idx_full.unsqueeze(-1).expand(-1, -1, D)    # [B, 1+k, D]
        x_reduced = x.gather(dim=1, index=idx_exp)            # [B, 1+k, D]

        return x_reduced, idx_full, scores


class RefinedTimmViT(nn.Module):
    """
    Wrap a timm VisionTransformer and insert token selection after warmup blocks.
    """
    def __init__(
        self,
        timm_name: str = "vit_base_patch16_224",
        pretrained: bool = False,
        num_classes:int = 10,
        warmup_depth: int = 2,
        keep_k: int = 64,
        score_hidden: int = 128,
    ):
        super().__init__()
        self.vit = timm.create_model(timm_name, pretrained=pretrained)
        self.vit.reset_classifier(num_classes=num_classes)

        self.warmup_depth = warmup_depth
        self.keep_k = keep_k

        # timm ViT has embed_dim at vit.embed_dim (usually)
        dim = self.vit.embed_dim
        self.selector = TopKTokenSelector(dim, hidden=score_hidden)

        # For convenience / debugging
        self.last_selected_idx = None
        self.last_scores = None

    def forward_features(self, x: torch.Tensor):
        vit = self.vit

        # ---- patchify ----
        x = vit.patch_embed(x)  # [B, N, D]
        B, N, D = x.shape

        # ---- add CLS + pos ----
        cls = vit.cls_token.expand(B, -1, -1)        # [B,1,D]
        x = torch.cat([cls, x], dim=1)               # [B,1+N,D]

        # pos_embed is [1, 1+N, D]
        if vit.pos_embed is not None:
            x = x + vit.pos_embed[:, : x.shape[1], :]
        x = vit.pos_drop(x)

        # ---- warmup SA blocks (contextual embeddings) ----
        for i in range(self.warmup_depth):
            x = vit.blocks[i](x)

        # ---- Step 3/4: score tokens and select top-k patches ----
        x_reduced, idx_full, scores = self.selector(x, k=self.keep_k)

        # (optional) keep for visualization
        self.last_selected_idx = idx_full.detach()
        self.last_scores = scores.detach()

        # ---- Continue remaining SA blocks on reduced tokens ----
        for i in range(self.warmup_depth, len(vit.blocks)):
            x_reduced = vit.blocks[i](x_reduced)

        # ---- norm + take CLS ----
        x_reduced = vit.norm(x_reduced)
        cls_out = x_reduced[:, 0]  # [B, D]
        return cls_out

    def forward(self, x: torch.Tensor):
        feat = self.forward_features(x)
        return self.vit.head(feat)