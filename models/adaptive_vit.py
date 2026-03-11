import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models.cross_tokens_selector import CrossTokenSelector, CrossTokenSelectorSaliency
from models.new_patch import OverlapPatchEmbed
from models.token_fuse import TokenFusion

class AdaptiveTokenVit(nn.Module):
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

        self.selector = CrossTokenSelectorSaliency(
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
        attended_patch = None

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
                selected_tokens, scores, attn, attended_patch, selected_idx = self.selector(
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
