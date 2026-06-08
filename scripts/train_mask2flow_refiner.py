#!/usr/bin/env python
"""Train Mask2Flow Phase A refiner from cached coarse STFT estimates."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sia_fm_tse.model import Mask2FlowCFM, Mask2FlowDiT


class Mask2FlowCache(Dataset):
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(self.cache_dir.glob("sample_*.pt"))

        if not self.files:
            raise RuntimeError(f"no cache files found in {self.cache_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        item = torch.load(self.files[idx], map_location="cpu")
        return {
            "mixture": item["mixture_stft"],
            "coarse": item["coarse_stft"],
            "target": item["target_stft"],
            "condition": item["condition"],
        }


def _read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _print_tensor(name: str, x: torch.Tensor) -> None:
    print(
        f"{name}: shape={tuple(x.shape)}, dtype={x.dtype}, "
        f"min={float(x.min().detach().cpu()):.6f}, "
        f"max={float(x.max().detach().cpu()):.6f}",
        flush=True,
    )


def main() -> None:
    parser = ArgumentParser(description="Train PN-conditioned Mask2Flow refiner.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("===== Mask2Flow refiner trainer =====", flush=True)
    print("config:", args.config, flush=True)
    print("dry_run:", args.dry_run, flush=True)

    conf = _read_yaml(args.config)

    if "paths" not in conf:
        raise RuntimeError("missing 'paths' section in config")
    if "model" not in conf:
        raise RuntimeError("missing 'model' section in config")
    if "train" not in conf:
        raise RuntimeError("missing 'train' section in config")

    paths = conf["paths"]
    model_conf = conf["model"]
    train_conf = conf["train"]

    cache_dir = Path(paths["cache_dir"])
    save_dir = Path(paths["save_dir"])

    print("cache_dir:", cache_dir, flush=True)
    print("save_dir:", save_dir, flush=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("device:", device, flush=True)

    dataset = Mask2FlowCache(cache_dir)
    print("cache samples:", len(dataset), flush=True)

    batch_size = int(train_conf["batch_size"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(train_conf.get("num_workers", 0)),
        drop_last=True,
    )

    print("batch_size:", batch_size, flush=True)
    print("train batches:", len(loader), flush=True)

    if len(loader) <= 0:
        raise RuntimeError(
            f"DataLoader has 0 batches. cache samples={len(dataset)}, batch_size={batch_size}, drop_last=True"
        )

    n_fft = int(model_conf["n_fft"])
    stft_dim = 2 * (n_fft // 2 + 1)

    print("n_fft:", n_fft, flush=True)
    print("stft_dim:", stft_dim, flush=True)
    print("model dim:", int(model_conf["dim"]), flush=True)
    print("depth:", int(model_conf["depth"]), flush=True)
    print("cond_in_ch:", int(model_conf.get("cond_in_ch", 64)), flush=True)
    print("cond_in_freq:", int(model_conf.get("cond_in_freq", 65)), flush=True)

    transformer = Mask2FlowDiT(
        dim=int(model_conf["dim"]),
        depth=int(model_conf["depth"]),
        n_head=int(model_conf["n_head"]),
        dim_head=int(model_conf["dim_head"]),
        dropout=float(model_conf["dropout"]),
        ff_mult=int(model_conf["ff_mult"]),
        stft_dim=stft_dim,
        cond_in_ch=int(model_conf.get("cond_in_ch", 64)),
        cond_in_freq=int(model_conf.get("cond_in_freq", 65)),
        long_skip_connection=bool(model_conf["long_skip_connection"]),
    )

    model = Mask2FlowCFM(
        transformer=transformer,
        cond_drop_prob=float(model_conf["cond_drop_prob"]),
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print("total params:", total_params, flush=True)
    print("trainable params:", trainable_params, flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_conf["lr"]),
        weight_decay=float(train_conf.get("weight_decay", 0.0)),
    )

    endpoint_l1_weight = float(train_conf.get("endpoint_l1_weight", 0.0))

    if args.dry_run:
        print("===== DRY RUN START =====", flush=True)

        batch = next(iter(loader))
        start = batch["coarse"].to(device)
        mixture = batch["mixture"].to(device)
        target = batch["target"].to(device)
        condition = batch["condition"].to(device)

        _print_tensor("start/coarse", start)
        _print_tensor("mixture", mixture)
        _print_tensor("target", target)
        _print_tensor("condition", condition)

        loss, metrics = model.loss(
            start,
            mixture,
            condition,
            target,
            endpoint_l1_weight=endpoint_l1_weight,
        )

        print("dry-run loss:", float(loss.detach().cpu()), flush=True)
        print(
            {
                k: float(v.detach().cpu())
                for k, v in metrics.items()
            },
            flush=True,
        )
        print("===== DRY RUN DONE =====", flush=True)
        return

    print("===== TRAIN START =====", flush=True)

    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(train_conf["epochs"])
    grad_clip = float(train_conf["grad_clip"])
    log_interval = int(train_conf.get("log_interval", 20))
    save_every_epoch = int(train_conf.get("save_every_epoch", 1))

    global_step = 0
    best_loss = float("inf")

    for epoch in range(epochs):
        total = 0.0
        count = 0

        iterator = tqdm(loader, desc=f"epoch {epoch + 1:03d}", dynamic_ncols=True)

        for batch in iterator:
            global_step += 1

            start = batch["coarse"].to(device)
            mixture = batch["mixture"].to(device)
            target = batch["target"].to(device)
            condition = batch["condition"].to(device)

            optimizer.zero_grad(set_to_none=True)

            loss, metrics = model.loss(
                start,
                mixture,
                condition,
                target,
                endpoint_l1_weight=endpoint_l1_weight,
            )

            if not torch.isfinite(loss):
                print(f"skip non-finite loss at step {global_step}: {loss.item()}", flush=True)
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total += loss.item()
            count += 1

            if global_step == 1 or global_step % log_interval == 0:
                iterator.set_postfix(
                    loss=f"{loss.item():.5f}",
                    flow=f"{metrics['flow_loss'].item():.5f}",
                    l1=f"{metrics['endpoint_l1'].item():.5f}",
                    grad=f"{float(grad_norm):.3f}",
                )

        epoch_loss = total / max(count, 1)
        print(f"epoch {epoch + 1:03d} | loss {epoch_loss:.6f}", flush=True)

        checkpoint = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": conf,
            "epoch_loss": epoch_loss,
        }

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(checkpoint, save_dir / "best.pt")
            print(f"saved best.pt | loss {best_loss:.6f}", flush=True)

        if save_every_epoch > 0 and (epoch + 1) % save_every_epoch == 0:
            out_path = save_dir / f"epoch_{epoch + 1:04d}.pt"
            torch.save(checkpoint, out_path)
            print(f"saved {out_path}", flush=True)

    print("===== TRAIN DONE =====", flush=True)


if __name__ == "__main__":
    main()
