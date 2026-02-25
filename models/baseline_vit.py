import timm
import torch
import torch.nn as nn


class BaselineTimmViT(nn.Module):
    """
    Plain timm ViT/DeiT baseline (no token pruning).
    Uses the same patch embed + blocks + norm + head as timm's VisionTransformer,
    but keeps the interface similar to your RefinedTimmViT.
    """
    def __init__(
        self,
        timm_name: str = "vit_base_patch16_224",
        #timm_name: str = "deit_tiny_patch16_224",
        pretrained: bool = False,
        num_classes: int = 10,
    ):
        super().__init__()
        self.vit = timm.create_model(timm_name, pretrained=pretrained)
        self.vit.reset_classifier(num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        return self.vit(x)

    # def forward_features(self, x: torch.Tensor) -> torch.Tensor:
    #     vit = self.vit
    #
    #     # patchify
    #     x = vit.patch_embed(x)  # [B, N, D]
    #     B, N, D = x.shape
    #
    #     # add CLS + pos
    #     cls = vit.cls_token.expand(B, -1, -1)  # [B,1,D]
    #     x = torch.cat([cls, x], dim=1)         # [B,1+N,D]
    #
    #     if vit.pos_embed is not None:
    #         x = x + vit.pos_embed[:, : x.shape[1], :]
    #     x = vit.pos_drop(x)
    #
    #     # run ALL blocks (no pruning)
    #     for blk in vit.blocks:
    #         x = blk(x)
    #
    #     x = vit.norm(x)
    #     return x[:, 0]  # CLS embedding
    #
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     feat = self.forward_features(x)
    #     return self.vit.head(feat)