import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class CrossTokenSelector_old(nn.Module):
    def __init__(self, embed_dim, num_heads=8, top_k=16, gumbel_tau=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.top_k = top_k
        self.gumbel_tau = gumbel_tau

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.score_proj = nn.Linear(embed_dim, 1)

    def forward(self, sampled_tokens, base_tokens, hard=True):
        """
        sampled_tokens: [B, Ns, C]
        base_tokens:    [B, Nb, C]
        """
        B, Ns, C = sampled_tokens.shape
        Nb = base_tokens.shape[1]
        H = self.num_heads
        Dh = C // H

        q = self.q_proj(sampled_tokens).reshape(B, Ns, H, Dh).transpose(1, 2)  # [B,H,Ns,Dh]
        k = self.k_proj(base_tokens).reshape(B, Nb, H, Dh).transpose(1, 2)     # [B,H,Nb,Dh]
        v = self.v_proj(base_tokens).reshape(B, Nb, H, Dh).transpose(1, 2)     # [B,H,Nb,Dh]

        attn = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)   # [B,H,Ns,Nb]
        attn = attn.softmax(dim=-1)

        refined = attn @ v                               # [B,H,Ns,Dh]
        refined = refined.transpose(1, 2).reshape(B, Ns, C)
        refined = self.out_proj(refined)                 # [B,Ns,C]

        # token importance scores
        scores = self.score_proj(refined).squeeze(-1)    # [B,Ns]

        if hard:
            # hard top-k selection
            topk_idx = scores.topk(k=min(self.top_k, Ns), dim=-1).indices
            selected = torch.gather(
                refined,
                1,
                topk_idx.unsqueeze(-1).expand(-1, -1, C)
            )
            return selected, scores, attn

        # soft / relaxed selection with gumbel
        weights = F.gumbel_softmax(scores, tau=self.gumbel_tau, hard=False, dim=-1)  # [B,Ns]
        weighted_tokens = refined * weights.unsqueeze(-1)
        topk_idx = scores.topk(k=min(self.top_k, Ns), dim=-1).indices
        selected = torch.gather(
            weighted_tokens,
            1,
            topk_idx.unsqueeze(-1).expand(-1, -1, C)
        )
        return selected, scores, attn


# learned embeddings are queries
# q = patch_toks

# k,v = sampled_tokens
class CrossTokenSelector(nn.Module):
    def __init__(self, embed_dim, num_heads=8, top_k=16, gumbel_tau=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.top_k = top_k
        self.gumbel_tau = gumbel_tau

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, base_tokens, sampled_tokens, hard=True):
        """
        base_tokens:    [B, Nb, C]  -> block-4 patch tokens (queries)
        sampled_tokens: [B, Ns, C]  -> sampled image tokens (keys/values)
        """
        B, Nb, C = base_tokens.shape
        Ns = sampled_tokens.shape[1]
        H = self.num_heads
        Dh = C // H

        q = self.q_proj(base_tokens).reshape(B, Nb, H, Dh).transpose(1, 2)      # [B,H,Nb,Dh]
        k = self.k_proj(sampled_tokens).reshape(B, Ns, H, Dh).transpose(1, 2)   # [B,H,Ns,Dh]
        v = self.v_proj(sampled_tokens).reshape(B, Ns, H, Dh).transpose(1, 2)   # [B,H,Ns,Dh]

        attn = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)   # [B,H,Nb,Ns]
        attn = attn.softmax(dim=-1)

        # updated base tokens if you want them for debugging
        attended = attn @ v                               # [B,H,Nb,Dh]
        attended = attended.transpose(1, 2).reshape(B, Nb, C)
        attended = self.out_proj(attended)                # [B,Nb,C]

        # score sampled tokens by how much the block-4 queries attend to them
        scores = attn.mean(dim=1).mean(dim=1)             # [B,Ns]
        # mean over heads, then mean over query tokens

        k_keep = min(self.top_k, Ns)
        topk_idx = scores.topk(k=k_keep, dim=-1).indices  # [B,k]

        selected = torch.gather(
            sampled_tokens,
            1,
            topk_idx.unsqueeze(-1).expand(-1, -1, C)
        )  # [B,k,C]

        if hard:
            return selected, scores, attn, attended, topk_idx

        # optional soft weighting before gathering
        weights = F.gumbel_softmax(scores, tau=self.gumbel_tau, hard=False, dim=-1)  # [B,Ns]
        weighted_sampled = sampled_tokens * weights.unsqueeze(-1)
        selected = torch.gather(
            weighted_sampled,
            1,
            topk_idx.unsqueeze(-1).expand(-1, -1, C)
        )
        return selected, scores, attn, attended, topk_idx