# # METHOD-WISE PSEUDOCODE (Steps 1–7)
# # ---------------------------------
# # Keeps your design choices explicit:
# #  - Q from Step-2 embeddings
# #  - K from Step-1 patches
# #  - V has 3 options: from patches / from embeddings / none (deferred)
# #  - routing has 2 options: discrete indices / continuous coordinates
# import timm
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class DynamicRoutingViT:
#     def __init__(self,
#                  patch_size=16,
#                  grid_hw=(14, 14),
#                  D=768,
#                  d_qk=128,
#                  M=32,
#                  num_classes=1000):
#
#         self.P = patch_size
#         self.Gh, self.Gw = grid_hw
#         self.N = self.Gh * self.Gw
#
#         self.D = D
#         self.d = d_qk
#         self.M = M
#
#         # --- Step 1 params ---
#         self.W_patch = Param((3 * self.P * self.P, self.D))
#         self.b_patch = Param((self.D,))
#         self.PosEmbed = Param((self.N, self.D))
#         self.CLS = Param((1, self.D))
#
#         # --- Step 3 params (router projections) ---
#         self.W_q = Param((self.D, self.d))
#         self.W_k = Param((self.D, self.d))
#         self.W_v = Param((self.D, self.D))  # keep V in model dim
#
#         # --- SA stacks ---
#         self.SA_early = TransformerEncoder(n_layers=4, D=self.D)
#         self.SA_late  = TransformerEncoder(n_layers=8, D=self.D)
#
#         # --- Step 7 head ---
#         self.W_head = Param((self.D, num_classes))
#         self.b_head = Param((num_classes,))
#
#         # Precomputed coords for continuous routing
#         self.patch_coords = self._make_patch_coords(self.Gh, self.Gw)  # (N,2)
#
#     # ============================================================
#     # Step 1) Input image -> patches -> tokens
#     # ============================================================
#     def step1_patch_embed(self, img):
#         """
#         img: (B, 3, 224, 224)
#         returns:
#           X0: (B, N=196, D=768)
#         """
#         patches = patchify(img, patch_size=self.P)              # (B, N, 3, P, P)
#         flat = patches.reshape(B, self.N, 3 * self.P * self.P)  # (B, N, 768)
#         X0 = linear(flat, self.W_patch, self.b_patch)           # (B, N, D)
#         X0 = X0 + self.PosEmbed[None, :, :]                     # (B, N, D)
#         return X0
#
#     # ============================================================
#     # Step 2) Early SA stack -> contextual embeddings
#     # ============================================================
#     def step2_contextualize(self, X0):
#         """
#         X0: (B, N, D)
#         returns:
#           cls: (B, 1, D)
#           H0 : (B, N, D)  # contextual patch embeddings
#         """
#         X_in = concat(self.CLS[None, :, :].repeat(B, axis=0), X0, axis=1)  # (B, 1+N, D)
#         H = self.SA_early(X_in)                                            # (B, 1+N, D)
#         cls = H[:, 0:1, :]                                                 # (B, 1, D)
#         H0  = H[:, 1:, :]                                                  # (B, N, D)
#         return cls, H0
#
#     # ============================================================
#     # Step 3) Build Q from Step-2 embeddings, K from Step-1 patches
#     #         + 3 options for V
#     # ============================================================
#     def step3_build_QKV(self, X0, H0, q_source="H0", V_mode="from_patches"):
#         """
#         X0: (B, N, D) from Step 1
#         H0: (B, N, D) from Step 2
#
#         q_source:
#           - "H0"  : Q from contextual patch embeddings (B, N, d)
#           - "CLS" : Q from CLS (requires passing cls separately; see step3_alt below)
#
#         V_mode:
#           - "from_patches"    : V from X0
#           - "from_embeddings" : V from H0
#           - "none"            : no V here (deferred sampling later)
#
#         returns:
#           Q: (B, Qn, d)
#           K: (B, N,  d)
#           V: (B, N,  D) or None
#         """
#         # Q from learned embeddings (Step 2)
#         if q_source == "H0":
#             Q = linear(H0, self.W_q)     # (B, N, d)
#         else:
#             raise ValueError("q_source must be 'H0' here; use step3_build_QKV_from_CLS for CLS")
#
#         # K from patches (Step 1)
#         K = linear(X0, self.W_k)         # (B, N, d)
#
#         # V options
#         if V_mode == "from_patches":
#             V = linear(X0, self.W_v)     # (B, N, D)
#         elif V_mode == "from_embeddings":
#             V = linear(H0, self.W_v)     # (B, N, D)
#         elif V_mode == "none":
#             V = None
#         else:
#             raise ValueError("Unknown V_mode")
#
#         return Q, K, V
#
#     def step3_build_QKV_from_CLS(self, X0, cls, H0, V_mode="from_patches"):
#         """
#         Same as above, but Q is only from CLS:
#           Q: (B, 1, d)
#         Useful if you want one global sampler.
#         """
#         Q = linear(cls, self.W_q)         # (B, 1, d)
#         K = linear(X0, self.W_k)          # (B, N, d)
#
#         if V_mode == "from_patches":
#             V = linear(X0, self.W_v)
#         elif V_mode == "from_embeddings":
#             V = linear(H0, self.W_v)
#         elif V_mode == "none":
#             V = None
#         else:
#             raise ValueError("Unknown V_mode")
#
#         return Q, K, V
#
#     # ============================================================
#     # Step 4) Convert (Q,K) -> sampling decision (discrete or continuous)
#     # ============================================================
#     def step4_route(self, Q, K, routing_mode="discrete", tau=1.0, k_select=32):
#         """
#         Q: (B, Qn, d)
#         K: (B, N,  d)
#
#         routing_mode:
#           - "discrete": output patch indices
#           - "continuous": output coordinates (between patches)
#
#         returns:
#           route:
#             if discrete   -> patch_idx: (B, M) or (B, Qn) depending on setup
#             if continuous -> coords   : (B, Qn, 2)
#           plus optional weights A if you want them
#         """
#         # S = QK^T / sqrt(d)
#         S = matmul(Q, K.transpose(-1, -2)) / sqrt(self.d)   # (B, Qn, N)
#
#         if routing_mode == "discrete":
#             # Pick patches by TopK on scores
#             # Common choice: if Qn==1 (CLS) then choose M=k_select patches
#             # If Qn>1, you can choose 1 per query (k=1) or K per query.
#             if Q.shape[1] == 1:
#                 _, idx = topk(S[:, 0, :], k=k_select, axis=-1)  # (B, M)
#                 return {"patch_idx": idx, "S": S}
#             else:
#                 # one patch per query (simple)
#                 _, idx = topk(S, k=1, axis=-1)                  # (B, Qn, 1)
#                 idx = idx.squeeze(-1)                           # (B, Qn)
#                 return {"patch_idx": idx, "S": S}
#
#         elif routing_mode == "continuous":
#             # Soft weights -> expected coordinate
#             A = softmax(S, axis=-1, temperature=tau)            # (B, Qn, N)
#             coords = self.patch_coords                           # (N,2)
#             p = matmul(A, coords[None, :, :].repeat(B, axis=0))  # (B, Qn, 2)
#             return {"coords": p, "A": A, "S": S}
#
#         else:
#             raise ValueError("Unknown routing_mode")
#
#     # ============================================================
#     # Step 5) Build re-sampled tokens (depends on V and routing_mode)
#     # ============================================================
#     def step5_resample_tokens(self, X0, H0, route, V, routing_mode="discrete", V_mode="from_patches"):
#         """
#         If V exists: resample from V.
#         If V_mode == "none": defer and sample from (X0 or H0) here anyway.
#
#         returns:
#           T: (B, M', D)
#         """
#         # Decide which source to sample from if V is None (deferred)
#         if V is None:
#             # choose default: sample from H0 (context) or X0 (raw)
#             # you said either can work; pick one:
#             V_src = H0
#         else:
#             V_src = V  # (B, N, D)
#
#         if routing_mode == "discrete":
#             idx = route["patch_idx"]               # (B, M') or (B, Qn)
#             T = gather_tokens(V_src, idx)          # (B, M', D)
#             return T
#
#         elif routing_mode == "continuous":
#             coords = route["coords"]               # (B, Qn, 2)
#             V_grid = V_src.reshape(B, self.Gh, self.Gw, self.D)  # (B,14,14,D)
#             T = bilinear_sample(V_grid, coords)    # (B, Qn, D)
#             return T
#
#         else:
#             raise ValueError("Unknown routing_mode")
#
#     # ============================================================
#     # Step 6) Remaining layers on re-sampled tokens (+ CLS)
#     # ============================================================
#     def step6_late_transformer(self, cls, T):
#         """
#         cls: (B,1,D)
#         T  : (B,M',D)
#         returns:
#           H_late: (B,1+M',D)
#         """
#         X_late_in = concat(cls, T, axis=1)         # (B, 1+M', D)
#         H_late = self.SA_late(X_late_in)           # (B, 1+M', D)
#         return H_late
#
#     # ============================================================
#     # Step 7) Task head (CLS)
#     # ============================================================
#     def step7_head(self, H_late):
#         """
#         H_late: (B,1+M',D)
#         returns:
#           y: (B,num_classes)
#         """
#         rep = H_late[:, 0, :]                      # (B,D)
#         y = linear(rep, self.W_head, self.b_head)  # (B,num_classes)
#         return y
#
#     # ============================================================
#     # Full forward (ties steps together)
#     # ============================================================
#     def forward(self,
#                 img,
#                 q_source="CLS",          # "CLS" or "H0"
#                 V_mode="from_patches",   # "from_patches" | "from_embeddings" | "none"
#                 routing_mode="continuous",# "discrete" | "continuous"
#                 tau=0.5,
#                 k_select=32):
#
#         # Step 1
#         X0 = self.step1_patch_embed(img)           # (B,N,D)
#
#         # Step 2
#         cls, H0 = self.step2_contextualize(X0)     # cls:(B,1,D), H0:(B,N,D)
#
#         # Step 3
#         if q_source == "CLS":
#             Q, K, V = self.step3_build_QKV_from_CLS(X0, cls, H0, V_mode=V_mode)  # Q:(B,1,d)
#         else:
#             Q, K, V = self.step3_build_QKV(X0, H0, q_source="H0", V_mode=V_mode) # Q:(B,N,d)
#
#         # Step 4
#         route = self.step4_route(Q, K, routing_mode=routing_mode, tau=tau, k_select=k_select)
#
#         # Step 5
#         T = self.step5_resample_tokens(X0, H0, route, V, routing_mode=routing_mode, V_mode=V_mode)
#
#         # Step 6
#         H_late = self.step6_late_transformer(cls, T)
#
#         # Step 7
#         y = self.step7_head(H_late)
#
#         return y
#
#     # ============================================================
#     # helpers
#     # ============================================================
#     def _make_patch_coords(self, Gh, Gw):
#         """
#         coords in normalized [-1,1] with centers aligned to patch order.
#         returns: (N,2) where coord[j] = (x,y)
#         """
#         coords = []
#         for y in range(Gh):
#             for x in range(Gw):
#                 # center coordinates
#                 # normalize to [-1,1]
#                 xn = (x + 0.5) / Gw * 2 - 1
#                 yn = (y + 0.5) / Gh * 2 - 1
#                 coords.append((xn, yn))
#         return Tensor(coords)  # (N,2)