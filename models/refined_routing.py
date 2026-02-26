# import timm
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# # not right
# class RoutedTimmViT(nn.Module):
#     """
#     timm ViT + routing between early and late blocks.
#
#     Routing design (your spec):
#       - Q derived from Step-2 embeddings H0 (contextual tokens)
#       - K derived from Step-1 patch tokens X0 (patch embeddings with pos)
#       - V can be:
#           (A) patches (X0)
#           (B) learned embeddings (H0)
#           (C) not required (return indices only)  [optional path]
#     """
#     def __init__(
#         self,
#         timm_name: str = "vit_base_patch16_224",
#         pretrained: bool = False,
#         num_classes: int = 10,
#         split_block: int = 4,         # how many early SA blocks to run before routing
#         num_routed: int = 32,         # M: number of routed tokens to keep
#         qk_dim: int = 128,            # d: Q/K projection dim
#         v_source: str = "h0",         # "x0" | "h0" | "none"
#         add_routed_pos: bool = True,  # optional pos for routed tokens (since they no longer map to grid)
#     ):
#         super().__init__()
#         self.vit = timm.create_model(timm_name, pretrained=pretrained)
#         self.vit.reset_classifier(num_classes=num_classes)
#
#         # Basic checks / assumptions (standard ViT)
#         assert hasattr(self.vit, "patch_embed"), "Expected a timm VisionTransformer-like model."
#         assert hasattr(self.vit, "blocks"), "Expected transformer blocks."
#         assert hasattr(self.vit, "pos_embed"), "Expected pos_embed."
#
#         self.split_block = split_block
#         self.num_routed = num_routed
#         self.qk_dim = qk_dim
#         self.v_source = v_source
#
#         D = self.vit.embed_dim
#         self.D = D
#
#         # Learned pooling from H0 patch tokens -> M query tokens (still Q-from-H0)
#         # P: (M, Npatch) will be softmaxed over patches
#         # We'll initialize later once we know Npatch from pos_embed shape.
#         n_tokens = self.vit.pos_embed.shape[1]  # usually 197 = 1 cls + 196 patches
#         self.n_patches = n_tokens - 1
#
#         self.pool_logits = nn.Parameter(torch.randn(num_routed, self.n_patches) * 0.02)
#
#         # Projections
#         self.Wq = nn.Linear(D, qk_dim, bias=False)
#         self.Wk = nn.Linear(D, qk_dim, bias=False)
#
#         if v_source in ("x0", "h0"):
#             self.Wv = nn.Linear(D, D, bias=False)
#         else:
#             self.Wv = None
#
#         # Optional: learned positional embeddings for routed tokens
#         self.add_routed_pos = add_routed_pos
#         if add_routed_pos:
#             self.routed_pos = nn.Parameter(torch.zeros(1, num_routed, D))
#             nn.init.trunc_normal_(self.routed_pos, std=0.02)
#         else:
#             self.routed_pos = None
#
#     def _patchify_embed(self, x: torch.Tensor):
#         """
#         Step 1 in timm style:
#           patch_embed -> add cls -> add pos -> pos_drop
#         Returns:
#           x_full: (B, 1+Npatch, D)
#           x0_patches: (B, Npatch, D)  [Step-1 patch tokens with pos]
#         """
#         B = x.shape[0]
#         x = self.vit.patch_embed(x)  # (B, Npatch, D)
#
#         # Add CLS token
#         cls = self.vit.cls_token.expand(B, -1, -1)  # (B,1,D)
#         x = torch.cat((cls, x), dim=1)              # (B,1+Npatch,D)
#
#         # Add pos embed (assumes input is same resolution as pretrained pos grid)
#         x = x + self.vit.pos_embed
#         x = self.vit.pos_drop(x)
#
#         x0_patches = x[:, 1:, :]  # Step-1 patches (with pos)
#         return x, x0_patches
#
#     def _run_early_blocks(self, x_full: torch.Tensor):
#         """
#         Step 2: early transformer blocks.
#         Returns H0_full: (B, 1+Npatch, D)
#         """
#         x = x_full
#         for blk in self.vit.blocks[: self.split_block]:
#             x = blk(x)
#         return x
#
#     def _routing(self, H0_full: torch.Tensor, X0_patches: torch.Tensor):
#         """
#         Steps 3–5:
#           Q from H0 embeddings (via learned pooling over patch tokens)
#           K from X0 patches
#           V: x0 or h0 or none
#
#         Returns:
#           T: (B, M, D) routed tokens (if v_source != "none")
#           or idx: (B, M) if v_source == "none" (hard indices)
#         """
#         B = H0_full.shape[0]
#         H0_patches = H0_full[:, 1:, :]  # (B, Npatch, D)
#
#         # --- Build M query tokens FROM H0 (your constraint), via learned pooling ---
#         # pool weights: (M, Npatch) -> softmax over patches
#         P = F.softmax(self.pool_logits, dim=-1)  # (M, Npatch)
#         # Q_src = P @ H0_patches  -> (B, M, D)
#         Q_src = torch.einsum("mn,bnd->bmd", P, H0_patches)
#
#         # Q from Step-2 learned embeddings
#         Q = self.Wq(Q_src)              # (B, M, d)
#
#         # K from Step-1 patches
#         K = self.Wk(X0_patches)         # (B, Npatch, d)
#
#         # Scores
#         S = torch.matmul(Q, K.transpose(-1, -2)) / (self.qk_dim ** 0.5)  # (B, M, Npatch)
#
#         if self.v_source == "none":
#             # "V not required": return patch indices (hard discrete routing)
#             # (This is non-differentiable; you can replace with gumbel/topk ST if desired.)
#             idx = torch.topk(S, k=1, dim=-1).indices.squeeze(-1)  # (B, M)
#             return idx
#
#         # Soft routing weights
#         A = F.softmax(S, dim=-1)  # (B, M, Npatch)
#
#         # Choose V
#         if self.v_source == "x0":
#             V = self.Wv(X0_patches)     # (B, Npatch, D)
#         elif self.v_source == "h0":
#             V = self.Wv(H0_patches)     # (B, Npatch, D)
#         else:
#             raise ValueError(f"Unknown v_source={self.v_source}")
#
#         # Routed tokens
#         T = torch.matmul(A, V)          # (B, M, D)
#
#         if self.add_routed_pos:
#             T = T + self.routed_pos     # (B, M, D)
#
#         return T
#
#     def _run_late_blocks_and_head(self, cls_from_H0: torch.Tensor, T_or_idx):
#         """
#         Step 6–7:
#           - build reduced sequence [CLS, routed_tokens] and run remaining blocks
#           - norm + head
#         """
#         if self.v_source == "none":
#             # idx: gather from *something* for the late stage; pick H0 patches by default
#             # Here we just re-use CLS and gather patch tokens from H0_full would require it passed in.
#             raise NotImplementedError(
#                 "v_source='none' returns indices only. Decide what you want to gather (X0 or H0), "
#                 "then implement gather here (and optionally positional encoding)."
#             )
#
#         T = T_or_idx  # (B, M, D)
#         x = torch.cat([cls_from_H0, T], dim=1)  # (B, 1+M, D)
#
#         # Remaining blocks only on reduced tokens
#         for blk in self.vit.blocks[self.split_block:]:
#             x = blk(x)
#
#         x = self.vit.norm(x)
#         cls = x[:, 0]                 # (B, D)
#         out = self.vit.head(cls)      # (B, num_classes)
#         return out
#
#     def forward(self, x: torch.Tensor):
#         # Step 1
#         x_full, X0_patches = self._patchify_embed(x)
#
#         # Step 2
#         H0_full = self._run_early_blocks(x_full)
#         cls_from_H0 = H0_full[:, :1, :]  # (B,1,D)
#
#         # Steps 3–5
#         routed = self._routing(H0_full, X0_patches)
#
#         # Steps 6–7
#         out = self._run_late_blocks_and_head(cls_from_H0, routed)
#         return out
#
# model = RoutedTimmViT(
#     timm_name="vit_base_patch16_224",
#     pretrained=False,
#     num_classes=10,
#     split_block=4,
#     num_routed=32,
#     qk_dim=128,
#     v_source="h0",      # "x0" or "h0"
#     add_routed_pos=True
# )
#
#
# import timm
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# def reduce_scores(S: torch.Tensor, reduce: str, dim: int) -> torch.Tensor:
#     """
#     S: (..., N, N) or (..., N, M) etc
#     reduce along `dim` to get a scalar per remaining index.
#     """
#     if reduce == "max":
#         return S.max(dim=dim).values
#     if reduce == "mean":
#         return S.mean(dim=dim)
#     if reduce == "logsumexp":
#         return torch.logsumexp(S, dim=dim)
#     raise ValueError(f"Unknown reduce='{reduce}' (use max/mean/logsumexp)")
#
#
# class TopKRouteBlock(nn.Module):
#     """
#     Implements Step 3-5 for either:
#       - mode='tokens': produce scores per *token i* (rows), TopK tokens
#       - mode='patches': produce scores per *patch j* (cols), TopK patches
#
#     Uses:
#       Q = H0 Wq
#       K = X0 Wk
#       S = Q K^T / sqrt(d)
#
#     Then:
#       - collapse S to 196 scores
#       - idx = topk(scores, K)
#       - gather from either H0 or X0 (configurable)
#     """
#     def __init__(
#         self,
#         embed_dim: int,
#         qk_dim: int = 128,
#         topk: int = 32,
#         mode: str = "tokens",          # "tokens" | "patches"
#         reduce: str = "logsumexp",     # "max" | "mean" | "logsumexp"
#         gather_from: str = "h0",       # for mode="patches": "x0" or "h0"; for mode="tokens" it's always "h0"
#     ):
#         super().__init__()
#         assert mode in ("tokens", "patches")
#         assert gather_from in ("x0", "h0")
#         self.mode = mode
#         self.reduce = reduce
#         self.topk = topk
#         self.gather_from = gather_from
#
#         self.Wq = nn.Linear(embed_dim, qk_dim, bias=False)
#         self.Wk = nn.Linear(embed_dim, qk_dim, bias=False)
#         self.scale = qk_dim ** -0.5
#
#     def forward(self, X0_patches: torch.Tensor, H0_patches: torch.Tensor):
#         """
#         X0_patches: (B, N, D) from Step 1 (after pos)
#         H0_patches: (B, N, D) from Step 2 (after early blocks)
#         Returns:
#           selected_tokens: (B, K, D)
#           idx: (B, K) indices in [0..N-1]
#           scores: (B, N) collapsed scores (for debugging)
#         """
#         B, N, D = H0_patches.shape
#         Kkeep = min(self.topk, N)
#
#         Q = self.Wq(H0_patches)            # (B, N, d)
#         K = self.Wk(X0_patches)            # (B, N, d)
#
#         # S: (B, N, N) token-to-patch affinity
#         S = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
#
#         if self.mode == "tokens":
#             # score each token i: collapse over patches j (columns)
#             scores = reduce_scores(S, self.reduce, dim=-1)        # (B, N)
#             idx = torch.topk(scores, k=Kkeep, dim=-1).indices     # (B, K)
#             # gather from contextual tokens (H0)
#             selected = torch.gather(
#                 H0_patches,
#                 dim=1,
#                 index=idx.unsqueeze(-1).expand(-1, -1, D)
#             )                                                     # (B, K, D)
#             return selected, idx, scores
#
#         else:
#             # mode == "patches"
#             # score each patch j: collapse over tokens i (rows)
#             scores = reduce_scores(S, self.reduce, dim=-2)        # (B, N)
#             idx = torch.topk(scores, k=Kkeep, dim=-1).indices     # (B, K)
#
#             source = X0_patches if self.gather_from == "x0" else H0_patches
#             selected = torch.gather(
#                 source,
#                 dim=1,
#                 index=idx.unsqueeze(-1).expand(-1, -1, D)
#             )                                                     # (B, K, D)
#             return selected, idx, scores
#
#
# class TimmViTWithTopKRouting(nn.Module):
#     """
#     timm ViT wrapper:
#       Step 1: patch embed + cls + pos
#       Step 2: run first `split_block` blocks -> H0
#       Step 3-5: TopKRouteBlock (tokens OR patches)
#       Step 6: run remaining blocks on [CLS + selected]
#       Step 7: norm + head (CLS)
#     """
#     def __init__(
#         self,
#         timm_name: str = "vit_base_patch16_224",
#         pretrained: bool = False,
#         num_classes: int = 10,
#         split_block: int = 4,
#         qk_dim: int = 128,
#         topk: int = 32,
#         mode: str = "tokens",          # "tokens" | "patches"
#         reduce: str = "logsumexp",     # "max" | "mean" | "logsumexp"
#         gather_from: str = "h0",       # for mode="patches": "x0" or "h0"
#     ):
#         super().__init__()
#         self.vit = timm.create_model(timm_name, pretrained=pretrained)
#         self.vit.reset_classifier(num_classes=num_classes)
#
#         assert hasattr(self.vit, "patch_embed")
#         assert hasattr(self.vit, "cls_token")
#         assert hasattr(self.vit, "pos_embed")
#         assert hasattr(self.vit, "pos_drop")
#         assert hasattr(self.vit, "blocks")
#         assert hasattr(self.vit, "norm")
#         assert hasattr(self.vit, "head")
#
#         self.split_block = split_block
#         D = self.vit.embed_dim
#
#         self.router = TopKRouteBlock(
#             embed_dim=D,
#             qk_dim=qk_dim,
#             topk=topk,
#             mode=mode,
#             reduce=reduce,
#             gather_from=gather_from,
#         )
#
#     def forward(self, x: torch.Tensor, return_debug: bool = False):
#         B = x.shape[0]
#
#         # ---- Step 1: patches -> tokens + cls + pos
#         x = self.vit.patch_embed(x)                         # (B, N, D)
#         cls = self.vit.cls_token.expand(B, -1, -1)          # (B, 1, D)
#         x = torch.cat((cls, x), dim=1)                      # (B, 1+N, D)
#         x = x + self.vit.pos_embed
#         x = self.vit.pos_drop(x)
#
#         X0_patches = x[:, 1:, :]                            # (B, N, D)
#
#         # ---- Step 2: early blocks
#         h = x
#         for blk in self.vit.blocks[: self.split_block]:
#             h = blk(h)
#
#         cls_h0 = h[:, :1, :]                                # (B, 1, D)
#         H0_patches = h[:, 1:, :]                            # (B, N, D)
#
#         # ---- Step 3-5: route + TopK + gather
#         selected, idx, scores = self.router(X0_patches, H0_patches)  # (B,K,D), (B,K), (B,N)
#
#         # ---- Step 6: remaining blocks on reduced tokens
#         h2 = torch.cat([cls_h0, selected], dim=1)            # (B, 1+K, D)
#         for blk in self.vit.blocks[self.split_block:]:
#             h2 = blk(h2)
#
#         # ---- Step 7: norm + head on CLS
#         h2 = self.vit.norm(h2)
#         out = self.vit.head(h2[:, 0])
#
#         if return_debug:
#             return out, {"idx": idx, "scores": scores}
#         return out
#
# model = TimmViTWithTopKRouting(
#     timm_name="vit_base_patch16_224",
#     pretrained=False,
#     num_classes=10,
#     split_block=4,
#     qk_dim=128,
#     topk=32,
#     mode="tokens",          # <-- Top-K tokens
#     reduce="logsumexp",
# )
#
# model = TimmViTWithTopKRouting(
#     timm_name="vit_base_patch16_224",
#     pretrained=False,
#     num_classes=10,
#     split_block=4,
#     qk_dim=128,
#     topk=32,
#     mode="patches",         # <-- Top-K patches
#     reduce="logsumexp",
#     gather_from="x0",       # "x0" (raw) or "h0" (contextual)
# )
#
# x = torch.randn(2, 3, 224, 224)
# y, dbg = model(x, return_debug=True)
# print(y.shape)        # (2, num_classes)
# print(dbg["idx"].shape, dbg["scores"].shape)  # (2, topk), (2, 196)