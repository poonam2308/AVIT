import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torch
import torch.nn as nn
import timm


class SimpleGateAdaptiveTokenVit(nn.Module):
    def __init__(
        self,
        model_name="vit_base_patch16_224",
        pretrained=True,
        num_classes=10,
        inject_after=3,          # after block 4 (0-based)
        dense_patch_size=16,
        dense_stride=4,
        img_size=224,
        cross_attn_heads=1,
    ):
        super().__init__()

        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

        self.embed_dim = self.vit.embed_dim
        self.inject_after = inject_after
        self.num_features = self.vit.num_features

        # Dense path
        self.dense_proj = nn.Conv2d(
            3,
            self.embed_dim,
            kernel_size=dense_patch_size,
            stride=dense_stride,
        )

        dense_grid = (img_size - dense_patch_size) // dense_stride + 1
        self.dense_grid = dense_grid
        self.num_dense_tokens = dense_grid * dense_grid

        self.pos_embed_dense = nn.Parameter(
            torch.zeros(1, self.num_dense_tokens, self.embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed_dense, std=0.02)

        # Cross-attention fusion
        self.cross_attn = nn.MultiheadAttention(
            self.embed_dim,
            num_heads=cross_attn_heads,
            batch_first=True,
        )

        # learnable gate: starts near zero so dense branch does not dominate early
        self.fusion_gate = nn.Parameter(torch.tensor(0.0))

        # transformer-style fusion block
        self.norm_fusion1 = nn.LayerNorm(self.embed_dim)
        self.norm_fusion2 = nn.LayerNorm(self.embed_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.GELU(),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
        )

    def forward_features(self, img, return_aux=False):
        # Standard ViT path
        x = self.vit.patch_embed(img)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)

        aux = {}

        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)

            if i == self.inject_after:
                # Split cls + patch tokens
                cls_tok = x[:, :1, :]      # [B,1,C]
                patch_toks = x[:, 1:, :]   # [B,N,C]

                # Dense path from raw image
                dense_feat = self.dense_proj(img)                    # [B,C,Hd,Wd]
                dense_feat = dense_feat.flatten(2).transpose(1, 2)   # [B,Nd,C]
                dense_feat = dense_feat + self.pos_embed_dense       # [B,Nd,C]

                # Only request weights when needed for visualization/debugging
                if return_aux:
                    attn_out, attn_weights = self.cross_attn(
                        query=patch_toks,
                        key=dense_feat,
                        value=dense_feat,
                        need_weights=True,
                        average_attn_weights=False,
                    )
                else:
                    attn_out, attn_weights = self.cross_attn(
                        query=patch_toks,
                        key=dense_feat,
                        value=dense_feat,
                        need_weights=False,
                    )

                # gated residual + MLP
                fused_patch_toks = self.norm_fusion1(
                    patch_toks + self.fusion_gate * attn_out
                )
                fused_patch_toks = self.norm_fusion2(
                    fused_patch_toks + self.fusion_mlp(fused_patch_toks)
                )

                # Put CLS back
                x = torch.cat([cls_tok, fused_patch_toks], dim=1)

                if return_aux:
                    aux = {
                        "dense_feat": dense_feat,
                        "attn_weights": attn_weights,   # [B, heads, N_patch, N_dense]
                        "attn_out": attn_out,
                        "fusion_gate": self.fusion_gate.detach(),
                        "patch_tokens_after_fusion": fused_patch_toks,
                    }

        x = self.vit.norm(x)
        cls_feat = x[:, 0]

        if return_aux:
            aux["tokens"] = x
            aux["cls_feat"] = cls_feat
            return cls_feat, aux

        return cls_feat

    def forward(self, x, return_aux=False):
        if return_aux:
            cls_feat, aux = self.forward_features(x, return_aux=True)
            logits = self.vit.head(cls_feat)
            aux["logits"] = logits
            return aux

        cls_feat = self.forward_features(x, return_aux=False)
        logits = self.vit.head(cls_feat)
        return logits