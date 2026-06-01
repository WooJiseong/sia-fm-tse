from __future__ import annotations

import argparse
import os
import traceback

import torch
import torchaudio
import wandb
import yaml
from tqdm import tqdm

from sia_fm_tse.bridge.PNAttentionFlow import ModelConfig, PNAttentionFlow
from sia_fm_tse.flowse.model.model_utils import get_tokenizer
from sia_fm_tse.pnenroll.dataset.LibriSpeech_single_emb import LibriDataset_single_emb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PNAttentionFlow")

    # ── Mode ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--train-mode",
        choices=["cfm_only", "end_to_end"],
        default="end_to_end",
        help=(
            "cfm_only: enrollment branch frozen, only CFM/DiT `trained "
            "(requires --enrollment-ckpt). "
            "end_to_end: entire model trained jointly."
        ),
    )

    # ── Paths ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--config",
        default="config/train.yaml",
        help="Path to train.yaml",
    )
    parser.add_argument(
        "--enrollment-ckpt",
        default="pn_baseline/proposed-monaural.pt",
        help="Path to pre-trained enrollment branch checkpoint (Tar_Model format). "
        "If not provided in end_to_end mode, enrollment branch is trained from scratch.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a PNAttentionFlow checkpoint to resume from.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory to save checkpoints. Defaults to output/{train_mode}.",
    )

    # ── Dataset ────────────────────────────────────────────────────────────────
    parser.add_argument("--train-dataset-dir", required=True)
    parser.add_argument("--val-dataset-dir", required=True)
    parser.add_argument("--noise-dir", type=str, default=None)
    parser.add_argument("--brir-dir", type=str, default=None)
    parser.add_argument("--wave-length", type=int, default=48000)
    parser.add_argument("--source-num", type=int, default=2)
    parser.add_argument("--min-source-num", type=int, default=1)
    parser.add_argument(
        "--active-num",
        type=int,
        nargs=2,
        default=[1, 2],
        metavar=("MIN", "MAX"),
        help="Range of active (target) speaker counts per sample.",
    )
    parser.add_argument(
        "--snr-db-range",
        type=float,
        nargs=2,
        default=[-5.0, 5.0],
        metavar=("MIN", "MAX"),
    )

    # ── ModelConfig (enrollment branch) ───────────────────────────────────────
    parser.add_argument("--head-layer-num", type=int, default=4)
    parser.add_argument("--head-refine-layer-num", type=int, default=2)
    parser.add_argument(
        "--head-fusion-shortcut",
        type=int,
        nargs="+",
        default=[0],
        metavar="IDX",
        help="Layer indices in enrollment head that use residual addition.",
    )

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--lr",
        type=float,
        default=7.5e-6,
        help="LR for main branch. Enrollment branch uses lr * 0.1 in end_to_end mode.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--warmup-steps", type=int, default=2500)
    parser.add_argument("--num-workers", type=int, default=10)

    # ── W&B ───────────────────────────────────────────────────────────────────
    parser.add_argument("--wandb-project", default="pn-attention-flowse")
    parser.add_argument(
        "--wandb-name",
        default=None,
        help="W&B run name. Defaults to {train_mode}_{pid}.",
    )
    parser.add_argument("--wandb-offline", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    process_id = os.getpid()
    checkpoint_dir = args.checkpoint_dir or f"output/{args.train_mode}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] train_mode: {args.train_mode}  device: {device}")

    # ── Config ────────────────────────────────────────────────────────────────
    with open(args.config) as f:
        train_conf = yaml.safe_load(f)

    model_conf = train_conf["model"]
    arch = dict(model_conf["arch"])
    mel_conf = dict(model_conf["mel_spec"])
    target_sr: int = mel_conf["target_sample_rate"]
    mel_dim: int = mel_conf["n_mel_channels"]

    tokenizer_path: str = model_conf["tokenizer_path"]  # Emilia_ZH_EN_pinyin/vocab.txt
    tokenizer: str = model_conf["tokenizer"]  # pinyin
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)
    text_num_embeds: int = vocab_size

    print(
        f"text_num_embeds: {text_num_embeds}  mel_dim: {mel_dim}  target_sr: {target_sr}"
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    # LibriDataset_single_emb returns (audio, pos_separated, neg_separated) where:
    #   audio:         [B, source_num, n_channels, n_samples]  — full mixture sources
    #   pos_separated: [B, active_num, n_channels, n_samples]  — target-speaker frames,
    #                  non-target frames zeroed out (same length as wave_length)
    #   neg_separated: [B, active_num, n_channels, n_samples]  — non-target frames,
    #                  target frames zeroed out
    sample_rate = 16000

    train_dataset = LibriDataset_single_emb(
        args.train_dataset_dir,
        sample_rate=sample_rate,
        wave_length=args.wave_length,
        pos_example_length=args.wave_length,
        neg_example_length=args.wave_length,
        snr_db_range=args.snr_db_range,
        source_num=args.source_num,
        min_source_num=args.min_source_num,
        active_num=args.active_num,
        normalize=False,
        reproducable=False,
        return_dvec=False,
        binaural=True,
        return_clean_dvec=False,
        noise_dir=os.path.join(args.noise_dir, "tr/"),
        brir_dir=args.brir_dir,
    )
    val_dataset = LibriDataset_single_emb(
        args.val_dataset_dir,
        sample_rate=sample_rate,
        wave_length=args.wave_length,
        pos_example_length=args.wave_length,
        neg_example_length=args.wave_length,
        snr_db_range=args.snr_db_range,
        source_num=args.source_num,
        min_source_num=args.min_source_num,
        active_num=args.active_num,
        normalize=False,
        return_dvec=False,
        binaural=True,
        return_clean_dvec=False,
        noise_dir=os.path.join(args.noise_dir, "cv/"),
        brir_dir=args.brir_dir,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    # ── ModelConfig ───────────────────────────────────────────────────────────
    conf = ModelConfig(
        # STFT / frequency — must match TFGridNet_encoder settings
        n_freqs=65,  # n_fft // 2 + 1 = 128 // 2 + 1
        n_fft=128,
        emb_dim=64,
        n_channels=2,  # binaural
        # Enrollment encoder
        enc_stride=64,
        enc_n_blocks=3,
        # Enrollment head
        head_layer_num=args.head_layer_num,
        head_refine_layer_num=args.head_refine_layer_num,
        head_fusion_shortcut=args.head_fusion_shortcut,
        head_cut_pos=True,
        # DiT — sourced directly from train.yaml arch block
        dit_dim=arch["dim"],
        dit_depth=arch["depth"],
        dit_heads=arch["heads"],
        dit_dim_head=arch.get("dim_head", 64),
        dit_dropout=arch.get("dropout", 0.1),
        dit_ff_mult=arch["ff_mult"],
        dit_mel_dim=mel_dim,  # must equal n_freqs
        dit_text_num_embeds=text_num_embeds,
        dit_text_dim=arch.get("text_dim"),
        dit_conv_layers=arch.get("conv_layers", 0),
        dit_long_skip_connection=arch.get("long_skip_connection", False),
        dit_checkpoint_activations=arch.get("checkpoint_activations", False),
        # CFM
        cfm_audio_drop_prob=model_conf.get("audio_drop_prob", 0.0),
        cfm_cond_drop_prob=model_conf.get("cond_drop_prob", 0.0),
        cfm_mel_spec_kwargs=mel_conf,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PNAttentionFlow(conf=conf, vocab_char_map=vocab_char_map).to(device)

    # Load enrollment branch from checkpoint if provided.
    # - provided + cfm_only:   load and freeze → only CFM/DiT trains
    # - provided + end_to_end: load and keep trainable → full fine-tune
    # - None     + end_to_end: skip → full training from scratch
    if args.enrollment_ckpt is not None:
        PNAttentionFlow.load_enrollment_branch(
            model,
            args.enrollment_ckpt,
            freeze=(args.train_mode == "cfm_only"),
            strict=False,
            device=device,
        )
    else:
        assert args.train_mode == "end_to_end", (
            "cfm_only mode requires --enrollment-ckpt"
        )
        print(
            "[main] no enrollment checkpoint — enrollment branch trained from scratch"
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"params  trainable: {n_trainable:,} / total: {n_total:,}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # In end_to_end mode the enrollment branch uses a 10x lower lr to avoid
    # destabilising the pre-trained (or randomly initialised) weights.
    if args.train_mode == "end_to_end":
        enrollment_params = list(model.enrollment_encoder.parameters()) + list(
            model.enrollment_head.parameters()
        )
        enrollment_param_ids = {id(p) for p in enrollment_params}
        other_params = [
            p for p in trainable_params if id(p) not in enrollment_param_ids
        ]
        param_groups: list[dict] = [
            {"name": "enrollment", "params": enrollment_params, "lr": args.lr * 0.1},
            {"name": "main", "params": other_params, "lr": args.lr},
        ]
    else:
        param_groups = [{"name": "main", "params": trainable_params, "lr": args.lr}]

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    # ── LR Scheduler ─────────────────────────────────────────────────────────
    total_steps = len(train_loader) * args.epochs
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=args.warmup_steps,
    )
    decay_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=1e-8,
        total_iters=max(1, total_steps - args.warmup_steps),
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[args.warmup_steps],
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_loss = float("inf")
    global_step = 0

    if args.resume is not None:
        cpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(cpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(cpt["optim_state_dict"])
        scheduler.load_state_dict(cpt["scheduler_state_dict"])
        start_epoch = cpt["epoch"] + 1
        best_loss = cpt["best_loss"]
        global_step = cpt.get("global_step", 0)
        print(f"[main] resumed from epoch {start_epoch}, best_loss {best_loss:.4f}")

    # ── Resampler (dataset 16 kHz -> FlowSE mel spec target_sr) ──────────────
    resampler = torchaudio.transforms.Resample(
        orig_freq=sample_rate,
        new_freq=target_sr,
    ).to(device)

    # ── W&B ───────────────────────────────────────────────────────────────────
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or f"{args.train_mode}_{process_id}",
        config={
            **vars(args),
            "n_trainable": n_trainable,
            "n_total": n_total,
            "total_steps": total_steps,
            "mel_dim": mel_dim,
            "target_sr": target_sr,
        },
    )
    wandb.watch(model, log="gradients", log_freq=100)

    # ── Log file ──────────────────────────────────────────────────────────────
    log = open(os.path.join(checkpoint_dir, f"{process_id}.txt"), "a")
    for k, v in vars(args).items():
        log.write(f"{k}: {v}\n")
    log.flush()

    cpt: dict = {}

    try:
        for epoch in range(start_epoch, args.epochs):
            # ── Train ─────────────────────────────────────────────────────────
            model.train()
            train_loss_acc = 0.0

            titer = tqdm(
                train_loader,
                desc=f"[{args.train_mode}] epoch {epoch:03d} train",
                dynamic_ncols=True,
            )
            for audio, pos_separated, neg_separated in titer:
                global_step += 1

                audio = audio.to(device)  # [B, source_num, C, N]
                pos_separated = pos_separated.to(device)  # [B, active_num,  C, N]
                neg_separated = neg_separated.to(device)  # [B, active_num,  C, N]

                gt = audio[:, 0]  # [B, C, N]  ground-truth target
                mixture = audio.sum(dim=1)  # [B, C, N]
                positive = pos_separated.sum(
                    dim=1
                )  # [B, C, N]  target frames, rest zeroed
                negative = neg_separated.sum(
                    dim=1
                )  # [B, C, N]  non-target frames, rest zeroed

                # Resample 16 kHz -> target_sr; merge channel into batch dim
                # so torchaudio resampler receives a 2-D input.
                B, C, N = mixture.shape
                mixture_rs = resampler(mixture.reshape(B * C, N)).reshape(B, C, -1)
                positive_rs = resampler(positive.reshape(B * C, N)).reshape(B, C, -1)
                negative_rs = resampler(negative.reshape(B * C, N)).reshape(B, C, -1)
                gt_rs = resampler(gt.reshape(B * C, N)).reshape(B, C, -1)

                # CFM expects [B, n_samples] for raw waveform; average channels.
                clean = gt_rs.mean(dim=1)  # [B, n_samples]
                text = [" "] * B  # transcription not available at this stage

                loss, _, _ = model(
                    mixture=mixture_rs,
                    positive=positive_rs,
                    negative=negative_rs,
                    clean=clean,
                    text=text,
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, max_norm=args.grad_clip
                )
                optimizer.step()
                scheduler.step()

                loss_val = loss.item()
                train_loss_acc += loss_val

                titer.set_postfix(loss=f"{loss_val:.4f}")
                wandb.log(
                    {
                        "train/loss": loss_val,
                        "train/lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

            train_loss = train_loss_acc / len(train_loader)

            # ── Validation ────────────────────────────────────────────────────
            model.eval()
            val_loss_acc = 0.0

            with torch.no_grad():
                for audio, pos_separated, neg_separated in tqdm(
                    val_loader,
                    desc=f"[{args.train_mode}] epoch {epoch:03d} val",
                    dynamic_ncols=True,
                    leave=False,
                ):
                    audio = audio.to(device)
                    pos_separated = pos_separated.to(device)
                    neg_separated = neg_separated.to(device)

                    gt = audio[:, 0]
                    mixture = audio.sum(dim=1)
                    positive = pos_separated.sum(dim=1)
                    negative = neg_separated.sum(dim=1)

                    B, C, N = mixture.shape
                    mixture_rs = resampler(mixture.reshape(B * C, N)).reshape(B, C, -1)
                    positive_rs = resampler(positive.reshape(B * C, N)).reshape(
                        B, C, -1
                    )
                    negative_rs = resampler(negative.reshape(B * C, N)).reshape(
                        B, C, -1
                    )
                    gt_rs = resampler(gt.reshape(B * C, N)).reshape(B, C, -1)
                    clean = gt_rs.mean(dim=1)
                    text = [" "] * B

                    loss, _, _ = model(
                        mixture=mixture_rs,
                        positive=positive_rs,
                        negative=negative_rs,
                        clean=clean,
                        text=text,
                    )
                    val_loss_acc += loss.item()

            val_loss = val_loss_acc / len(val_loader)
            lr_now = optimizer.param_groups[0]["lr"]

            print(
                f"[epoch {epoch:03d}] "
                f"train: {train_loss:.4f}  val: {val_loss:.4f}  lr: {lr_now:.2e}"
            )
            log.write(
                f"epoch: {epoch}  train: {train_loss:.4f}  "
                f"val: {val_loss:.4f}  lr: {lr_now:.2e}\n"
            )
            log.flush()

            wandb.log(
                {
                    "epoch/train_loss": train_loss,
                    "epoch/val_loss": val_loss,
                    "epoch/lr": lr_now,
                    "epoch": epoch,
                },
                step=global_step,
            )

            # ── Checkpoint ────────────────────────────────────────────────────
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss

            cpt = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "train_mode": args.train_mode,
            }
            torch.save(cpt, os.path.join(checkpoint_dir, "last.pt"))
            if is_best:
                torch.save(cpt, os.path.join(checkpoint_dir, "best.pt"))
                print(f"[epoch {epoch:03d}] best saved (val_loss: {best_loss:.4f})")
                wandb.summary["best_val_loss"] = best_loss
                wandb.summary["best_epoch"] = epoch

    except KeyboardInterrupt:
        print("[main] interrupted — saving checkpoint")
        if cpt:
            torch.save(
                cpt, os.path.join(checkpoint_dir, f"interrupted_{process_id}.pt")
            )
    except Exception:
        print("[main] training failed:")
        print(traceback.format_exc())
    finally:
        log.close()
        wandb.finish()


if __name__ == "__main__":
    main()
