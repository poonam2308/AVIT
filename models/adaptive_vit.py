import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models.cross_tokens_selector import CrossTokenSelector, CrossTokenSelectorSaliency, CrossTokenSelectorSaliency_new
from models.new_patch import OverlapPatchEmbed
from models.refined_sampling_block import RefinedSamplingBlock
from models.token_fuse import TokenFusion

class AdaptiveTokenVit_old(nn.Module):
    def __init__(
        self,
        model_name="vit_base_patch16_224",
        pretrained=False,
        num_classes=10,
        inject_after=3,          # 0-based => after 4th block
        overlap_patch_size=16,
        overlap_stride=1,
        top_k=16,
    ):
        super().__init__()

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

        self.embed_dim = self.model.embed_dim
        self.inject_after = inject_after
        self.num_features = self.model.num_features

        self.overlap_embed = OverlapPatchEmbed(
            in_chans=3,
            patch_size=overlap_patch_size,
            stride=overlap_stride,
            embed_dim=self.embed_dim,
        )

        self.selector = CrossTokenSelectorSaliency_new(
            embed_dim=self.embed_dim,
            num_heads=self.model.blocks[0].attn.num_heads,
            top_k=top_k,
        )

        # self.fuse = nn.Linear(self.embed_dim * 2, self.embed_dim)

        self.fuse = TokenFusion(
            dim=self.embed_dim,
            heads=self.model.blocks[0].attn.num_heads,
        )

    def forward_features(self, x, return_aux=False):
        B = x.shape[0]

        # standard timm patch tokens
        x_patch = self.model.patch_embed(x)   # [B, N, C]

        cls_token = self.model.cls_token.expand(B, -1, -1)
        if self.model.pos_embed is not None:
            x_tokens = torch.cat((cls_token, x_patch), dim=1)
            x_tokens = x_tokens + self.model.pos_embed[:, : x_tokens.shape[1], :]
        else:
            x_tokens = torch.cat((cls_token, x_patch), dim=1)

        x_tokens = self.model.pos_drop(x_tokens)

        sampled_hw = None
        sampled_pos = None
        scores = None
        attn = None
        selected_tokens = None
        fused_patch_toks = None
        selected_idx = None

        # run first 4 blocks
        for i, blk in enumerate(self.model.blocks):
            x_tokens = blk(x_tokens)

            if i == self.inject_after:
                # split cls and patch tokens
                cls_tok = x_tokens[:, :1, :]
                patch_toks = x_tokens[:, 1:, :]

                # candidate overlapping patch tokens from original image
                # sampled_tokens, sampled_hw = self.overlap_embed(x)
                # overlapping sampled tokens + 2D sin-cos positional embeddings
                sampled_tokens, sampled_hw, sampled_pos = self.overlap_embed.forward_with_pos(x)

                # sampled_tokens attend to block-4 patch tokens
                # selected_tokens, scores, attn = self.selector(
                #     sampled_tokens=sampled_tokens,
                #     base_tokens=patch_toks,
                #     hard=True,
                # )
                # queries are block-4 patch tokens, keys/values are sampled image patch tokens
                selected_tokens, scores, attn, selected_idx, _ = self.selector(
                    base_tokens=patch_toks,
                    sampled_tokens=sampled_tokens,
                    hard=False,
                )

                # simple fusion: concatenate selected sampled tokens to patch tokens
                # fused_patch_toks = torch.cat([patch_toks, selected_tokens], dim=1)
                fused_patch_toks = self.fuse(patch_toks, selected_tokens)

                # bring back cls token
                x_tokens = torch.cat([cls_tok, fused_patch_toks], dim=1)

        x_tokens = self.model.norm(x_tokens)
        cls_feat = x_tokens[:, 0]

        if return_aux:
            aux = {
                "sampled_hw": sampled_hw,
                "sampled_pos": sampled_pos,
                "selector_scores": scores,
                "selector_attn": attn,
                "selected_tokens": selected_tokens,
                "patch_tokens_after_fusion": fused_patch_toks,
                "tokens": x_tokens,
                "cls_feat": cls_feat,
                "selected_idx": selected_idx,
            }
            return cls_feat, aux

        return cls_feat

    def forward(self, x, return_aux=False):
        if return_aux:
            cls_feat, aux = self.forward_features(x, return_aux=True)
            logits = self.model.head(cls_feat)
            aux["logits"] = logits
            return aux

        cls_feat = self.forward_features(x, return_aux=False)
        logits = self.model.head(cls_feat)
        return logits


# adaptive_vit.py

class AdaptiveTokenVit(nn.Module):
    def __init__(
            self,
            model_name="vit_base_patch16_224",
            pretrained=True,
            num_classes=10,
            inject_after=3,
            overlap_patch_size=16,
            overlap_stride=4,  # Optimized stride
            top_k=100,
    ):
        super().__init__()

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

        self.embed_dim = self.model.embed_dim
        self.inject_after = inject_after

        # Integrate the RefinedSamplingBlock directly
        self.refiner = RefinedSamplingBlock(
            embed_dim=self.embed_dim,
            patch_size=overlap_patch_size,
            top_k=top_k,
            stride=overlap_stride
        )

    def forward_features(self, x):
        B = x.shape[0]

        # 1. Initial Patch Embedding
        x_patch = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(B, -1, -1)

        if self.model.pos_embed is not None:
            x_tokens = torch.cat((cls_token, x_patch), dim=1)
            x_tokens = x_tokens + self.model.pos_embed[:, : x_tokens.shape[1], :]
        else:
            x_tokens = torch.cat((cls_token, x_patch), dim=1)

        x_tokens = self.model.pos_drop(x_tokens)

        # 2. Sequential Block Processing
        for i, blk in enumerate(self.model.blocks):
            x_tokens = blk(x_tokens)

            # 3. Injection Point
            if i == self.inject_after:
                # Use the provided RefinedSamplingBlock code
                # This handles sampling, energy scoring, top-k selection, and concatenation
                x_tokens = self.refiner(x, x_tokens)

        x_tokens = self.model.norm(x_tokens)
        return x_tokens[:, 0]  # Return CLS token feature

    def forward(self, x):
        cls_feat = self.forward_features(x)
        logits = self.model.head(cls_feat)
        return logits




class SimpleAdaptiveTokenVit(nn.Module):
    def __init__(
        self,
        model_name="vit_base_patch16_224",
        pretrained=True,
        num_classes=10,
        inject_after=3,          # after block 4 (0-based)
        embed_dim=768,
        dense_patch_size=16,
        dense_stride=4,
        img_size=224,
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
            num_heads=1,
            batch_first=True,
        )
        self.norm_fusion = nn.LayerNorm(self.embed_dim)

    def forward_features(self, img, return_aux=False):
        # Standard ViT path
        x = self.vit.patch_embed(img)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)

        aux = {}

        # Run blocks and inject dense fusion after block 4
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

                # Cross-attention: patch tokens query dense image tokens
                attn_out, attn_weights = self.cross_attn(
                    query=patch_toks,
                    key=dense_feat,
                    value=dense_feat,
                    need_weights=True,
                    average_attn_weights=False,
                )

                fused_patch_toks = self.norm_fusion(patch_toks + attn_out)

                # Put CLS back
                x = torch.cat([cls_tok, fused_patch_toks], dim=1)

                if return_aux:
                    aux = {
                        "dense_feat": dense_feat,
                        "attn_weights": attn_weights,   # [B, heads, N_patch, N_dense]
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