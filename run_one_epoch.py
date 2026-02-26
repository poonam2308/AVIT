# run_one_epoch.py
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.route_gumbel_vit import diversity_batch_cosine, diversity_usage_uniform


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    global_step: int,
    epoch: int,
    grad_clip: Optional[float] = 1.0,
    log_every: int = 50,
    wandb_run=None,
    csv_writer=None,  # <-- NEW
) -> Tuple[int, Dict[str, float]]:
    model.train()

    # running sums for epoch averages
    n_logs = 0
    loss_sum = 0.0
    sums = {
        "loss": 0.0,
        "loss_task": 0.0,
        "loss_budget": 0.0,
        "loss_entropy": 0.0,
        "keep_ratio_mean": 0.0,
        "tau": 0.0,
    }

    for it, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        #out: Dict[str, torch.Tensor] = model(images, labels=labels, global_step=global_step)
        #loss = out["loss"]
        # logits = model(images)
        # loss = F.cross_entropy(logits, labels)

        # ---- if model has routing schedule, update tau + gumbel flag ----
        if hasattr(model, "routing_schedule") and hasattr(model, "router"):
            tau, use_gumbel = model.routing_schedule(global_step)
            if hasattr(model.router, "set_tau"):
                model.router.set_tau(tau)
            else:
                model.router.tau = float(tau)

            if hasattr(model.router, "set_use_gumbel"):
                model.router.set_use_gumbel(use_gumbel)
            elif hasattr(model.router, "use_gumbel"):
                model.router.use_gumbel = bool(use_gumbel)

        # ---- forward (request debug if supported) ----
        try:
            logits, dbg = model(images, return_debug=True)
        except TypeError:
            logits = model(images)
            dbg = None

        loss_task = F.cross_entropy(logits, labels)
        loss = loss_task

        # ---- diversity regularizer (optional) ----
        lambda_div = getattr(model, "lambda_div", 0.0)
        if lambda_div > 0.0 and dbg is not None and isinstance(dbg, dict) and "soft_w" in dbg:
            if getattr(model, "div_type", "usage_entropy") == "batch_cosine":
                loss_div = diversity_batch_cosine(dbg["soft_w"])
            else:
                loss_div = diversity_usage_uniform(dbg["soft_w"])
            loss = loss + lambda_div * loss_div
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        if it % log_every == 0:
            row = {
                "step": global_step,
                "epoch": epoch,
                "iter": it,
                "loss": float(loss.item()),
                # "loss_task": float(out["loss_task"].item()),
                # "loss_budget": float(out["loss_budget"].item()),
                # "loss_entropy": float(out["loss_entropy"].item()),
                # "keep_ratio_mean": float(out["keep_ratio_mean"].item()),
                # "tau": float(out["tau"].item()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "tau": float(getattr(getattr(model, "router", None), "tau", 0.0)),
            }
            print(f"step {global_step:6d}  loss {row['loss']:.4f}  lr {row['lr']:.2e}")

            # print(
            #     f"step {global_step:6d}  "
            #     f"loss {row['loss']:.4f}  "
            #     f"task {row['loss_task']:.4f}  "
            #     f"bud {row['loss_budget']:.4f}  "
            #     f"ent {row['loss_entropy']:.4f}  "
            #     f"keep {row['keep_ratio_mean']:.3f}  "
            #     f"tau {row['tau']:.3f}"
            # )

            # accumulate epoch averages (over logged points)
            # for k in sums:
            #     sums[k] += row[k]
            # n_logs += 1

            loss_sum += row["loss"]
            n_logs += 1

            # local CSV logging
            if csv_writer is not None:
                csv_writer.writerow(row)

            # wandb logging

            if wandb_run is not None:
                wandb_run.log({"train/loss": row["loss"], "train/lr": row["lr"],
                               "epoch": epoch}, step=global_step)

            # if wandb_run is not None:
            #     wandb_run.log(
            #         {
            #             "train/loss": row["loss"],
            #             "train/loss_task": row["loss_task"],
            #             "train/loss_budget": row["loss_budget"],
            #             "train/loss_entropy": row["loss_entropy"],
            #             "train/keep_ratio_mean": row["keep_ratio_mean"],
            #             "train/tau": row["tau"],
            #             "train/lr": row["lr"],
            #             "epoch": epoch,
            #         },
            #         step=global_step,
            #     )

        global_step += 1

    # return epoch averages (over log points)
    # if n_logs == 0:
    #     epoch_stats = {k: 0.0 for k in sums}
    # else:
    #     epoch_stats = {k: v / n_logs for k, v in sums.items()}
    #
    # return global_step, epoch_stats

    epoch_stats = {"loss": (loss_sum / max(1, n_logs))}
    return global_step, epoch_stats


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.numel()

    return correct / max(1, total)


import torch
import numpy as np
import matplotlib.pyplot as plt

@torch.no_grad()
def show_token_keep_mask(model, img_tensor, mean, std, title="Kept tokens"):
    """
    img_tensor: [1,3,H,W] normalized
    mean,std: lists/tuples of length 3 for de-normalization
    """
    model.eval()
    _ = model(img_tensor)  # forward populates last_selected_idx/last_scores

    vit = model.vit
    G = vit.patch_embed.grid_size[0]
    N = G * G

    idx_full = model.last_selected_idx[0].cpu().numpy()  # [1+k]
    # patch token indices are 1..N (0 is CLS)
    patch_idx = idx_full[idx_full != 0] - 1  # now 0..N-1

    mask = np.zeros(N, dtype=np.float32)
    mask[patch_idx] = 1.0
    mask = mask.reshape(G, G)

    # de-normalize image
    x = img_tensor[0].cpu()
    mean_t = torch.tensor(mean)[:, None, None]
    std_t  = torch.tensor(std)[:, None, None]
    x = (x * std_t + mean_t).clamp(0, 1)
    img = x.permute(1,2,0).numpy()  # HWC

    plt.figure()
    plt.imshow(img)
    # overlay mask as blocky heatmap
    plt.imshow(mask, alpha=0.20, interpolation="nearest",
               extent=(0, img.shape[1], img.shape[0], 0))
    plt.title(title)
    plt.axis("off")
    plt.show()

@torch.no_grad()
def show_token_score_heatmap(model, img_tensor, mean, std, title="Token scores"):
    model.eval()
    _ = model(img_tensor)

    vit = model.vit
    G = vit.patch_embed.grid_size[0]
    scores = model.last_scores[0].cpu().numpy()  # [1+N]
    patch_scores = scores[1:]                    # [N]
    hm = patch_scores.reshape(G, G)

    # de-normalize image
    x = img_tensor[0].cpu()
    mean_t = torch.tensor(mean)[:, None, None]
    std_t  = torch.tensor(std)[:, None, None]
    x = (x * std_t + mean_t).clamp(0, 1)
    img = x.permute(1,2,0).numpy()

    plt.figure()
    plt.imshow(img)
    plt.imshow(hm, alpha=0.45, interpolation="nearest",
               extent=(0, img.shape[1], img.shape[0], 0))
    plt.title(title)
    plt.axis("off")
    plt.show()

@torch.no_grad()
def avg_keep_probability_over_loader(model, loader, device):
    vit = model.vit
    G = vit.patch_embed.grid_size[0]
    N = G*G

    keep_counts = np.zeros(N, dtype=np.float64)
    total = 0

    model.eval()
    for imgs, _ in loader:
        imgs = imgs.to(device)
        _ = model(imgs)

        idx = model.last_selected_idx.cpu().numpy()  # [B, 1+k]
        for b in range(idx.shape[0]):
            patch_idx = idx[b][idx[b] != 0] - 1
            keep_counts[patch_idx] += 1
            total += 1

    keep_prob = (keep_counts / total).reshape(G, G)
    plt.figure()
    plt.imshow(keep_prob, interpolation="nearest")
    plt.title("Keep probability per patch position")
    plt.colorbar()
    plt.axis("off")
    plt.show()