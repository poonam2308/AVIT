import torch
import torch.nn as nn
import timm

import torch
import torch.nn as nn

class HeatmapPositionSelector(nn.Module):
    """
    Learns a spatial heatmap over patch grid (GxG) and selects top-k positions.
    """
    def __init__(self, embed_dim: int, grid_size: int, hidden: int = 128):
        super().__init__()
        self.grid_size = grid_size
        # 1x1 conv = per-location MLP on feature map
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x_tokens: torch.Tensor, k: int):
        """
        x_tokens: [B, 1+N, D] where N=G*G, CLS at 0
        returns:
          x_reduced: [B, 1+k, D]
          idx_full: [B, 1+k] token indices in original token sequence (CLS included)
          heatmap: [B, G, G]
        """
        B, Ntot, D = x_tokens.shape
        G = self.grid_size
        assert Ntot == 1 + G * G

        cls = x_tokens[:, :1, :]           # [B,1,D]
        patches = x_tokens[:, 1:, :]       # [B, G*G, D]

        # reshape patches to feature map
        fmap = patches.transpose(1, 2).reshape(B, D, G, G)  # [B,D,G,G]

        # heatmap scores
        hm = self.head(fmap).squeeze(1)                    # [B,G,G]
        flat = hm.flatten(1)                               # [B, G*G]

        k = min(k, G * G)
        topk = flat.topk(k, dim=1).indices                 # [B,k] in [0..G*G-1]

        # convert patch indices to token indices (shift by +1 because CLS is token 0)
        idx_full = torch.cat(
            [torch.zeros(B, 1, device=x_tokens.device, dtype=topk.dtype), topk + 1],
            dim=1
        )  # [B,1+k]

        idx_exp = idx_full.unsqueeze(-1).expand(-1, -1, D)
        x_reduced = x_tokens.gather(dim=1, index=idx_exp)

        return x_reduced, idx_full, hm

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

        # G = self.vit.patch_embed.grid_size[0]  # (G,G)
        # self.selector = HeatmapPositionSelector(dim, grid_size=G, hidden=score_hidden)


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

        # x_reduced, idx_full, heatmap = self.selector(x, k=self.keep_k)
        # self.last_scores = heatmap.detach()  # now it's a [B,G,G] heatmap
        # self.last_selected_idx = idx_full.detach()

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