import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import wandb
import yaml

from datasets import imagenet_style_loaders, cifar10_loaders
from models.baseline_vit import BaselineTimmViT
from models.gumbel_masked_vit import MaskedViTConfig, MaskedViT
from models.refined_vit import RefinedTimmViT
from models.route_gumbel_vit import TimmViTWithTopKRouting_STGumbel, RoutingSchedule
from models.routevit import RouteGumbelViTTokenReduction, RouteGumbelViTTokenEmphasis, \
    RouteGumbelViTTokenReductionConcat
from run_one_epoch import train_one_epoch, evaluate
from models.adaptive_vit import AdaptiveTokenVit


def deep_update(base: dict, updates: dict) -> dict:
    # shallow-but-useful for our structure (dataset/model/gating/gumbel/train/wandb)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def parse_overrides(override_list):
    """
    Accept overrides like:
      --override model.embed_dim=384
      --override gating.target_keep_ratio=0.5
    """
    out = {}
    for s in override_list or []:
        key, val = s.split("=", 1)
        keys = key.split(".")
        cur = out
        for kk in keys[:-1]:
            cur = cur.setdefault(kk, {})

        if val.lower() in ("true", "false"):
            parsed = val.lower() == "true"
        elif val.lower() in ("null", "none"):
            parsed = None
        else:
            try:
                parsed = int(val)
            except ValueError:
                try:
                    parsed = float(val)
                except ValueError:
                    parsed = val

        cur[keys[-1]] = parsed
    return out


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--override", action="append", default=[])
    return p.parse_args()


def save_checkpoint(path: Path, model, optimizer, cfg_dict: dict, epoch: int, global_step: int, best_acc: float):
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "best_acc": best_acc,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": cfg_dict,
        },
        str(path),
    )


def main():
    args = get_args()

    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    overrides = parse_overrides(args.override)
    cfg_dict = deep_update(cfg_dict, overrides)

    # --- wandb (optional) ---
    wandb_run = None
    wandb_cfg = cfg_dict.get("wandb", {})
    if wandb_cfg.get("enabled", False):
        wandb_run = wandb.init(
            project=wandb_cfg.get("project", "masked-vit"),
            entity=wandb_cfg.get("entity", None),
            name=wandb_cfg.get("name", None),
            tags=wandb_cfg.get("tags", None),
            config=cfg_dict,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dset = cfg_dict["dataset"]
    train_cfg = cfg_dict["train"]
    model_cfg = cfg_dict["model"]
    # gating = cfg_dict["gating"]
    # gumbel = cfg_dict["gumbel"]

    out_dir = Path(train_cfg.get("out_dir", "./outputs/run1"))
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # save resolved config
    with open(out_dir / "config_resolved.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # CSV logging
    csv_path = out_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    # fieldnames = [
    #     "step", "epoch", "iter",
    #     "loss", "loss_task", "loss_budget", "loss_entropy",
    #     "keep_ratio_mean", "tau", "lr",
    # ]
    fieldnames = ["step", "epoch", "iter", "loss", "lr", "tau"]
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    try:
        # data
        name = dset.get("name", "imagenet_style").lower()

        if name in ("cifar10", "cifar-10"):
            train_loader, val_loader = cifar10_loaders(
                data_root=dset.get("data_root", "./data"),
                batch_size=dset["batch_size"],
                num_workers=dset.get("num_workers", 4),
                img_size=dset.get("img_size", 224),
            )
        else:
            train_loader, val_loader = imagenet_style_loaders(
                data_root=dset["data_root"],
                img_size=dset["img_size"],
                batch_size=dset["batch_size"],
                num_workers=dset.get("num_workers", 4),
            )
        # train_loader, val_loader = imagenet_style_loaders(
        #     data_root=dset["data_root"],
        #     img_size=dset["img_size"],
        #     batch_size=dset["batch_size"],
        #     num_workers=dset.get("num_workers", 4),
        # )

        # model
        # vit_cfg = MaskedViTConfig(
        #     img_size=dset["img_size"],
        #     patch_size=model_cfg["patch_size"],
        #     in_chans=3,
        #     num_classes=dset["num_classes"],
        #     embed_dim=model_cfg["embed_dim"],
        #     depth=model_cfg["depth"],
        #     warmup_depth=model_cfg["warmup_depth"],
        #     num_heads=model_cfg["num_heads"],
        #     mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        #     dropout=model_cfg.get("dropout", 0.1),
        #     attn_dropout=model_cfg.get("attn_dropout", 0.1),
        #     gate_mlp=gating.get("gate_mlp", True),
        #     gate_hidden=gating.get("gate_hidden", 128),
        #     gate_mlp_updates=gating.get("gate_mlp_updates", True),
        #     target_keep_ratio=gating["target_keep_ratio"],
        #     lambda_budget=gating["lambda_budget"],
        #     lambda_entropy=gating["lambda_entropy"],
        #     tau_start=gumbel["tau_start"],
        #     tau_end=gumbel["tau_end"],
        #     tau_anneal_steps=gumbel["tau_anneal_steps"],
        #     hard_gates_train=gumbel.get("hard_gates_train", False),
        #     inference_topk=gumbel.get("inference_topk", None),
        # )

        # model = MaskedViT(vit_cfg).to(device)

        # model = RefinedTimmViT(
        #     timm_name=model_cfg.get("timm_name", "vit_base_patch16_224"),
        #     pretrained=model_cfg.get("pretrained", True),
        #     num_classes=dset["num_classes"],
        #     warmup_depth=model_cfg.get("warmup_depth", 2),
        #     keep_k=model_cfg.get("keep_k", 64),
        #     score_hidden=model_cfg.get("score_hidden", 128),
        # ).to(device)

        model_type = model_cfg.get("type", "refined").lower()

        if model_type == "baseline":
            print("Using BASELINE ViT")
            model = BaselineTimmViT(
                timm_name=model_cfg.get("timm_name", "vit_base_patch16_224"),
                pretrained=model_cfg.get("pretrained", True),
                num_classes=dset["num_classes"],
            ).to(device)


        elif model_type in ("route_gumbel", "route_gumbel_vit", "st_gumbel"):
            print("Using ROUTE GUMBEL ViT (ST Gumbel-TopK)")
            model = TimmViTWithTopKRouting_STGumbel(
                timm_name=model_cfg.get("timm_name", "deit_small_patch16_224"),
                pretrained=model_cfg.get("pretrained", True),
                num_classes=dset["num_classes"],
                split_block=model_cfg.get("split_block", 4),
                qk_dim=model_cfg.get("qk_dim", 128),
                keep_k=model_cfg.get("keep_k", 32),
                mode=model_cfg.get("mode", "tokens"),  # "tokens" or "patches"
                reduce=model_cfg.get("reduce", "logsumexp"),
                gather_from=model_cfg.get("gather_from", "h0"),  # if mode="patches": "x0" or "h0"
                tau=model_cfg.get("tau_start", 2.0),  # start tau (we'll anneal in train loop)
                add_routed_pos=model_cfg.get("add_routed_pos", True),
            ).to(device)

            model.routing_schedule = RoutingSchedule(
                tau0=model_cfg.get("tau_start", 2.0),
                tau_min=model_cfg.get("tau_end", 0.5),
                decay=model_cfg.get("tau_decay", 0.9995),
                gumbel_off_step=model_cfg.get("gumbel_off_step", 20000),
            )
            model.lambda_div = model_cfg.get("lambda_div", 0.0)
            model.div_type = model_cfg.get("div_type", "usage_entropy")


        elif model_type == "token_reduction":
            print("Using ROUTE GUMBEL ViT (ST Gumbel-TopK)")
            model = RouteGumbelViTTokenReductionConcat(
                timm_name=model_cfg.get("timm_name", "vit_base_patch16_224"),
                pretrained=model_cfg.get("pretrained", False),
                num_classes=dset["num_classes"],
                split_block=model_cfg.get("split_block", 4),
                keep_k=model_cfg.get("keep_k", 32)
            ).to(device)

        elif model_type == "token_emphasis":
            print("Using ROUTE GUMBEL ViT (ST Gumbel-TopK)")
            model = RouteGumbelViTTokenEmphasis(
                timm_name=model_cfg.get("timm_name", "vit_base_patch16_224"),
                pretrained=model_cfg.get("pretrained", False),
                num_classes=dset["num_classes"],
                split_block=model_cfg.get("split_block", 4),
                keep_k=model_cfg.get("keep_k", 32)
            ).to(device)

        elif model_type in ("adaptive_vit", "adaptive", "adaptive_token_vit"):
            print("Using ADAPTIVE ViT")

            model = AdaptiveTokenVit(
                model_name=model_cfg.get("timm_name", "vit_base_patch16_224"),
                pretrained=model_cfg.get("pretrained", False),
                num_classes=dset["num_classes"],
                inject_after=model_cfg.get("inject_after", 3),
                overlap_patch_size=model_cfg.get("overlap_patch_size", 16),
                overlap_stride=model_cfg.get("overlap_stride", 8),
                top_k=model_cfg.get("top_k", 16),
            ).to(device)

        else:
            print("Using REFINED ViT (token selection)")
            model = RefinedTimmViT(
                timm_name=model_cfg.get("timm_name", "vit_base_patch16_224"),
                pretrained=model_cfg.get("pretrained", True),
                num_classes=dset["num_classes"],
                warmup_depth=model_cfg.get("warmup_depth", 2),
                keep_k=model_cfg.get("keep_k", 64),
                score_hidden=model_cfg.get("score_hidden", 128),
            ).to(device)




        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg.get("weight_decay", 0.05),
        )

        best_acc = -1.0
        # history = {"epoch": [], "train_loss": [], "train_keep": [], "train_tau": [], "val_acc": []}
        history = {"epoch": [], "train_loss": [], "val_acc": []}
        global_step = 0
        for epoch in range(1, train_cfg["epochs"] + 1):
            global_step, epoch_stats = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                global_step,
                epoch=epoch,
                grad_clip=train_cfg.get("grad_clip", 1.0),
                log_every=train_cfg.get("log_every", 50),
                wandb_run=wandb_run,
                csv_writer=csv_writer,
            )

            acc = evaluate(model, val_loader, device)
            print(f"[eval] epoch {epoch}  acc {acc * 100:.2f}%")

            history["epoch"].append(epoch)
            history["train_loss"].append(epoch_stats["loss"])
            # history["train_keep"].append(epoch_stats["keep_ratio_mean"])
            # history["train_tau"].append(epoch_stats["tau"])
            history["val_acc"].append(acc)

            if wandb_run is not None:
                wandb_run.log({"val/acc": acc, "epoch": epoch}, step=global_step)

            # save "last"
            save_checkpoint(out_dir / "checkpoints" / "last.pth", model, optimizer, cfg_dict, epoch, global_step, best_acc)

            # save "best"
            if acc > best_acc:
                best_acc = acc
                save_checkpoint(out_dir / "checkpoints" / "best.pth", model, optimizer, cfg_dict, epoch, global_step, best_acc)
                print(f"[ckpt] new best acc {best_acc * 100:.2f}% -> saved best.pth")

        print("done")

        # plots
        plt.figure()
        plt.plot(history["epoch"], history["train_loss"])
        plt.xlabel("epoch")
        plt.ylabel("train loss (avg over log points)")
        plt.title("Training loss")
        plt.savefig(out_dir / "loss_curve.png", dpi=150)
        plt.close()

        # plt.figure()
        # plt.plot(history["epoch"], history["train_keep"], label="keep_ratio")
        # plt.plot(history["epoch"], history["train_tau"], label="tau")
        # plt.xlabel("epoch")
        # plt.title("Keep ratio & tau")
        # plt.legend()
        # plt.savefig(out_dir / "keep_tau.png", dpi=150)
        # plt.close()

        plt.figure()
        plt.plot(history["epoch"], history["val_acc"])
        plt.xlabel("epoch")
        plt.ylabel("val acc")
        plt.title("Validation accuracy")
        plt.savefig(out_dir / "val_acc.png", dpi=150)
        plt.close()

    finally:
        csv_file.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()