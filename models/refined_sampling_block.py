import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RefinedSamplingBlock(nn.Module):
    def __init__(self, embed_dim=768, patch_size=16, top_k=100, stride=4):
        super().__init__()
        self.patch_size = patch_size
        self.top_k = top_k
        self.stride = stride
        self.embed_dim = embed_dim

        # 1. Projections: Map raw pixels to ViT space
        self.q_proj = nn.Linear(patch_size * patch_size * 3, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 2. Normalization: Crucial for training stability
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_k = nn.LayerNorm(embed_dim)

        # 3. Output Projection: Fuses refined tokens back to model scale
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable gate (starts at 0)

    def forward(self, x_img, vit_features):
        """
        x_img: Original RGB Image [B, 3, 224, 224]
        vit_features: Output from ViT Block 4 [B, 197, 768]
        """
        B, N, C = vit_features.shape
        H_img, W_img = x_img.shape[-2:]

        # STEP 1: Dense Sampling (Stride-S)
        # Extract all possible overlapping patches
        patches = F.unfold(x_img, kernel_size=self.patch_size, stride=self.stride)
        patches = rearrange(patches, 'b c n -> b n c')  # [B, N_samples, 768]

        # STEP 2: Project to Embedding Space
        # q = Dense Samples, k = Existing ViT tokens (excluding CLS)
        q = self.norm_q(self.q_proj(patches))
        k = self.norm_k(self.k_proj(vit_features[:, 1:, :]))

        # STEP 3: Energy-based Saliency (Object Discovery)
        energy = torch.einsum('bid, bjd -> bij', q, k) / (self.embed_dim ** 0.5)
        scores = energy.max(dim=-1)[0]  # [B, N_samples]

        # STEP 4: Diversity Filtering (MaxPool)
        map_side = (H_img - self.patch_size) // self.stride + 1
        scores_2d = scores.view(B, 1, map_side, map_side)
        local_max = F.max_pool2d(scores_2d, kernel_size=3, stride=1, padding=1)
        keep_mask = (scores_2d == local_max).float()

        # Select Top-K from non-crowded candidates
        diversified_scores = (scores_2d * keep_mask).view(B, -1)
        _, topk_indices = torch.topk(diversified_scores, self.top_k, dim=1)

        # STEP 5: Gather & Refine the "Better Samples"
        # Extract the winners
        batch_idx = torch.arange(B).unsqueeze(-1).expand(-1, self.top_k).to(x_img.device)
        selected_patches = patches[batch_idx, topk_indices]  # [B, top_k, 768]

        # Refine through projection
        refined_tokens = self.out_proj(self.q_proj(selected_patches))

        # STEP 6: Concatenate for next Transformer Block
        # We append the 100 "Better Samples" to the 197 original tokens
        # We use gamma to let the model gradually learn to use these tokens
        combined_tokens = torch.cat([vit_features, self.gamma * refined_tokens], dim=1)

        return combined_tokens

# Example Usage in a Training Loop
# model = RefinedSamplingBlock(top_k=100, stride=4)
# next_layer_input = model(images, vit_block4_output)