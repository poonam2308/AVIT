import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

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