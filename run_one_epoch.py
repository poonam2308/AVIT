# run_one_epoch.py
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


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
        out: Dict[str, torch.Tensor] = model(images, labels=labels, global_step=global_step)
        loss = out["loss"]
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
                "loss_task": float(out["loss_task"].item()),
                "loss_budget": float(out["loss_budget"].item()),
                "loss_entropy": float(out["loss_entropy"].item()),
                "keep_ratio_mean": float(out["keep_ratio_mean"].item()),
                "tau": float(out["tau"].item()),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }

            print(
                f"step {global_step:6d}  "
                f"loss {row['loss']:.4f}  "
                f"task {row['loss_task']:.4f}  "
                f"bud {row['loss_budget']:.4f}  "
                f"ent {row['loss_entropy']:.4f}  "
                f"keep {row['keep_ratio_mean']:.3f}  "
                f"tau {row['tau']:.3f}"
            )

            # accumulate epoch averages (over logged points)
            for k in sums:
                sums[k] += row[k]
            n_logs += 1

            # local CSV logging
            if csv_writer is not None:
                csv_writer.writerow(row)

            # wandb logging
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": row["loss"],
                        "train/loss_task": row["loss_task"],
                        "train/loss_budget": row["loss_budget"],
                        "train/loss_entropy": row["loss_entropy"],
                        "train/keep_ratio_mean": row["keep_ratio_mean"],
                        "train/tau": row["tau"],
                        "train/lr": row["lr"],
                        "epoch": epoch,
                    },
                    step=global_step,
                )

        global_step += 1

    # return epoch averages (over log points)
    if n_logs == 0:
        epoch_stats = {k: 0.0 for k in sums}
    else:
        epoch_stats = {k: v / n_logs for k, v in sums.items()}

    return global_step, epoch_stats


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        out = model(images, labels=None, global_step=0)
        pred = out["logits"].argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.numel()

    return correct / max(1, total)