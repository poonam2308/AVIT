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
    def __init__(self, embed_dim, num_heads=8, top_k=32, gumbel_tau=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.top_k = top_k
        self.gumbel_tau = gumbel_tau

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.score_head = nn.Linear(embed_dim, 1)

    def forward(self, base_tokens, sampled_tokens, hard=True):
        """
        base_tokens:    [B, Nb, C]  -> block-4 patch tokens (queries)
        sampled_tokens: [B, Ns, C]  -> sampled image tokens (keys/values)
        """
        B, Nb, C = base_tokens.shape
        Ns = sampled_tokens.shape[1]
        H = self.num_heads
        Dh = C // H

        # queries from block-4 tokens
        q = self.q_proj(base_tokens).reshape(B, Nb, H, Dh).transpose(1, 2)      # [B,H,Nb,Dh]
        # keys/values from sampled tokens
        k = self.k_proj(sampled_tokens).reshape(B, Ns, H, Dh).transpose(1, 2)   # [B,H,Ns,Dh]
        v = self.v_proj(sampled_tokens).reshape(B, Ns, H, Dh).transpose(1, 2)   # [B,H,Ns,Dh]

        attn = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)   # [B,H,Nb,Ns]
        attn = attn.softmax(dim=-1)

        # updated base tokens if you want them for debugging
        attended = attn @ v                               # [B,H,Nb,Dh]
        attended = attended.transpose(1, 2).reshape(B, Nb, C)
        attended = self.out_proj(attended)                # [B,Nb,C]

        # Transformer-style refinement on query/output tokens
        x = self.norm1(base_tokens + attended)
        x = self.norm2(x + self.mlp(x))

        # score refined output tokens, not sampled tokens
        scores = self.score_head(x).squeeze(-1)  # [B,Nb]

        k_keep = min(self.top_k, Nb)
        topk_idx = scores.topk(k=k_keep, dim=-1).indices  # [B,k]

        selected = torch.gather(
            x,
            1,
            topk_idx.unsqueeze(-1).expand(-1, -1, C)
        )  # [B,k,C]

        if hard:
            return selected, scores, attn, x, topk_idx

        # optional soft weighting before gathering
        weights = F.gumbel_softmax(scores, tau=self.gumbel_tau, hard=False, dim=-1)  # [B,Ns]
        weighted_x = x * weights.unsqueeze(-1)
        selected = torch.gather(
            weighted_x,
            1,
            topk_idx.unsqueeze(-1).expand(-1, -1, C)
        )
        return selected, scores, attn, x, topk_idx


class CrossTokenSelectorSaliency(nn.Module):
    def __init__(self, embed_dim, num_heads=8, top_k=32, gumbel_tau=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.top_k = top_k
        self.gumbel_tau = gumbel_tau

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, base_tokens, sampled_tokens, hard=True):
        """
        base_tokens:   [B, Nb, C]  queries from block-4
        sampled_tokens:[B, Ns, C]  keys/values from new patches
        """
        B, Nb, C = base_tokens.shape
        Ns = sampled_tokens.shape[1]
        H = self.num_heads
        Dh = C // H

        q = self.q_proj(base_tokens).reshape(B, Nb, H, Dh).transpose(1, 2)      # [B,H,Nb,Dh]
        k = self.k_proj(sampled_tokens).reshape(B, Ns, H, Dh).transpose(1, 2)   # [B,H,Ns,Dh]
        v = self.v_proj(sampled_tokens).reshape(B, Ns, H, Dh).transpose(1, 2)   # [B,H,Ns,Dh]

        # raw attention logits
        logits = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)                         # [B,H,Nb,Ns]

        # score each query token BEFORE softmax
        query_scores = logits.max(dim=-1).values.mean(dim=1)                     # [B,Nb]

        k_keep = min(self.top_k, Nb)
        # topk_idx = query_scores.topk(k=k_keep, dim=-1).indices                   # [B,k]
        if hard:
            select_scores = query_scores
        else:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(query_scores).clamp_min(1e-9)))
            select_scores = query_scores + self.gumbel_tau * gumbel_noise

        topk_idx = select_scores.topk(k=k_keep, dim=-1).indices

        # gather selected query logits
        selected_logits = torch.gather(
            logits.transpose(1, 2),  # [B,Nb,H,Ns]
            1,
            topk_idx[:, :, None, None].expand(-1, -1, H, Ns)
        ).transpose(1, 2)            # [B,H,k,Ns]

        # softmax only after selecting top-k queries
        selected_attn = selected_logits.softmax(dim=-1)                          # [B,H,k,Ns]

        selected_out = selected_attn @ v                                          # [B,H,k,Dh]
        selected_out = selected_out.transpose(1, 2).reshape(B, k_keep, C)        # [B,k,C]
        selected_out = self.out_proj(selected_out)

        # transformer-style refinement on selected outputs
        selected_base = torch.gather(
            base_tokens,
            1,
            topk_idx.unsqueeze(-1).expand(-1, -1, C)
        )                                                                         # [B,k,C]

        x = self.norm1(selected_base + selected_out)
        x = self.norm2(x + self.mlp(x))                                           # [B,k,C]

        return x, query_scores, selected_attn, topk_idx



class CrossTokenSelectorSaliency_new(nn.Module):
    def __init__(self, embed_dim, num_heads=8, top_k=32, gumbel_tau=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.top_k = top_k
        self.gumbel_tau = gumbel_tau

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, base_tokens, sampled_tokens, hard=True):
        """
        base_tokens:    [B, Nb, C]  -> block-4 patch tokens
        sampled_tokens: [B, Ns, C]  -> dense sampled patch tokens

        Logic:
        1. score sampled tokens by max similarity with base tokens
        2. top-k sampled token selection
        3. use base_tokens as queries, selected sampled tokens as keys/values
        4. attention + MLP refinement on base tokens
        """
        B, Nb, C = base_tokens.shape
        Ns = sampled_tokens.shape[1]
        H = self.num_heads
        Dh = C // H

        # ---- Stage 1: prototype-style saliency scoring over sampled tokens ----
        # similarity between sampled patches and block-4 tokens
        # [B, Ns, Nb]
        saliency_logits = torch.einsum(
            "bid,bjd->bij", sampled_tokens, base_tokens
        ) / (C ** 0.5)

        # score each sampled token by its strongest match to any base token
        sampled_scores = saliency_logits.max(dim=-1).values   # [B, Ns]

        k_keep = min(self.top_k, Ns)

        if hard:
            select_scores = sampled_scores
        else:
            gumbel_noise = -torch.log(
                -torch.log(torch.rand_like(sampled_scores).clamp_min(1e-9))
            )
            select_scores = sampled_scores + self.gumbel_tau * gumbel_noise

        topk_idx = select_scores.topk(k=k_keep, dim=-1).indices   # [B, k]

        selected_sampled = torch.gather(
            sampled_tokens,
            1,
            topk_idx.unsqueeze(-1).expand(-1, -1, C)
        )   # [B, k, C]

        # ---- Stage 2: cross-attention using selected sampled tokens as K,V ----
        q = self.q_proj(base_tokens).reshape(B, Nb, H, Dh).transpose(1, 2)             # [B,H,Nb,Dh]
        k = self.k_proj(selected_sampled).reshape(B, k_keep, H, Dh).transpose(1, 2)    # [B,H,k,Dh]
        v = self.v_proj(selected_sampled).reshape(B, k_keep, H, Dh).transpose(1, 2)    # [B,H,k,Dh]

        attn_logits = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)                           # [B,H,Nb,k]
        attn = attn_logits.softmax(dim=-1)

        attended = attn @ v                                                              # [B,H,Nb,Dh]
        attended = attended.transpose(1, 2).reshape(B, Nb, C)                           # [B,Nb,C]
        attended = self.out_proj(attended)

        # transformer-style refinement on base tokens
        x = self.norm1(base_tokens + attended)
        x = self.norm2(x + self.mlp(x))                                                  # [B,Nb,C]

        return x, sampled_scores, attn, topk_idx, selected_sampled