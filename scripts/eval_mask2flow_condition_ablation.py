#!/usr/bin/env python
"""Condition ablation for Mask2Flow mixture-start refiner.

Tests whether Mask2Flow uses PN condition:
- correct
- zero_condition
- shuffled_condition
- zero_pos
- zero_neg
- swapped_pos_neg
"""

from __future__ import annotations

import csv
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchaudio
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from sia_fm_tse.model import Encoder, Mask2FlowCFM, Mask2FlowDiT
from sia_fm_tse.utils import LibriDataset, istft_torch, load_eval_config, stft_torch


def si_snr(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)
    ref_energy = torch.sum(reference ** 2, dim=-1, keepdim=True).clamp_min(eps)
    projection = torch.sum(estimate * reference, dim=-1, keepdim=True) * reference / ref_energy
    noise = estimate - projection
    ratio = torch.sum(projection ** 2, dim=-1) / torch.sum(noise ** 2, dim=-1).clamp_min(eps)
    return 10.0 * torch.log10(ratio.clamp_min(eps))


def l1_audio(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y).abs().mean(dim=-1)


def safe_audio(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().cpu()
    peak = x.abs().max().clamp_min(1e-8)
    if peak > 0.99:
        x = x / peak * 0.99
    return x


def save_wav(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), safe_audio(wav).unsqueeze(0), sample_rate)


class Mask2FlowCondAblationModel(nn.Module):
    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        encoder_ckpt_path: str | Path,
        n_fft: int,
        hop_length: int,
        win_length: int,
        steps: int,
        cfg_strength: float,
    ):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.steps = int(steps)
        self.cfg_strength = float(cfg_strength)

        self.encoder = Encoder()
        self.encoder.load(Path(encoder_ckpt_path))
        self.encoder.eval()

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model_conf: dict[str, Any] = ckpt["config"]["model"]

        stft_dim = 2 * (int(model_conf["n_fft"]) // 2 + 1)

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

        self.refiner = Mask2FlowCFM(
            transformer=transformer,
            cond_drop_prob=float(model_conf.get("cond_drop_prob", 0.0)),
        )
        self.refiner.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.refiner.eval()

    @torch.no_grad()
    def make_condition(self, pos: torch.Tensor, neg: torch.Tensor) -> dict[str, torch.Tensor]:
        correct = self.encoder(pos, neg)

        conditions = {
            "correct": correct,
            "zero_condition": torch.zeros_like(correct),
            "zero_pos": self.encoder(torch.zeros_like(pos), neg),
            "zero_neg": self.encoder(pos, torch.zeros_like(neg)),
            "swapped_pos_neg": self.encoder(neg, pos),
        }

        if correct.shape[0] > 1:
            conditions["shuffled_condition"] = correct.roll(shifts=1, dims=0)
        else:
            conditions["shuffled_condition"] = torch.zeros_like(correct)

        return conditions

    @torch.no_grad()
    def decode_with_condition(
        self,
        mixture: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        length = mixture.shape[-1]

        mixture_stft = stft_torch(
            mixture,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        mixture_stft = rearrange(mixture_stft, "b d t -> b t d")

        # mixture-start Mask2Flow
        start = mixture_stft

        refined_stft, _ = self.refiner(
            start=start,
            mixture=mixture_stft,
            c=condition,
            steps=self.steps,
            cfg_strength=self.cfg_strength,
        )

        refined_stft = rearrange(refined_stft, "b t d -> b d t")

        pred = istft_torch(
            refined_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            length=length,
        )
        return pred


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--noise_dir", required=True)
    parser.add_argument("--noise_split", default="tt", choices=["tr", "cv", "tt"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg_strength", type=float, default=0.0)
    parser.add_argument("--save_wavs", action="store_true")
    parser.add_argument("--save_samples", type=int, default=4)
    parser.add_argument("--out_dir", default="outputs/mask2flow_condition_ablation/mixture")
    args = parser.parse_args()

    conf = load_eval_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = Mask2FlowCondAblationModel(
        checkpoint_path=args.checkpoint,
        encoder_ckpt_path=conf.encoder_ckpt_path,
        n_fft=conf.n_fft,
        hop_length=conf.hop_length,
        win_length=conf.win_length,
        steps=args.steps,
        cfg_strength=args.cfg_strength,
    ).to(device)
    model.eval()

    noise_path = Path(args.noise_dir) / args.noise_split

    dataset = LibriDataset(
        args.data_dir,
        sample_rate=conf.sample_rate,
        wave_length=3 * conf.sample_rate,
        pos_example_length=3 * conf.sample_rate,
        neg_example_length=3 * conf.sample_rate,
        snr_db_range=conf.snr_db_range,
        min_source_num=conf.min_source_num,
        source_num=conf.source_num,
        active_num=conf.active_num,
        reproducable=True,
        normalize=False,
        filling_pattern="repeat",
        return_dvec=False,
        dvec_rate=50,
        include_silent=False,
        special_spk=[],
        reverb="none",
        binaural=False,
        reverb_cond=False,
        zero_in_tgt=False,
        noise_dir=str(noise_path) + "/",
        same_disturb=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=args.batch_size > 1,
    )

    variants = [
        "correct",
        "zero_condition",
        "shuffled_condition",
        "zero_pos",
        "zero_neg",
        "swapped_pos_neg",
    ]

    totals = {
        name: {
            "si_snr": 0.0,
            "si_snri": 0.0,
            "l1_target": 0.0,
            "l1_correct": 0.0,
            "l1_mixture": 0.0,
            "cond_l1": 0.0,
        }
        for name in variants
    }

    mixture_total = {"si_snr": 0.0, "l1_target": 0.0}
    rows = []
    count = 0
    saved = 0

    print("===== Mask2Flow condition ablation =====", flush=True)
    print("checkpoint:", args.checkpoint, flush=True)
    print("config:", args.config, flush=True)
    print("data_dir:", args.data_dir, flush=True)
    print("noise_dir:", str(noise_path) + "/", flush=True)
    print("steps:", args.steps, flush=True)
    print("cfg_strength:", args.cfg_strength, flush=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="mask2flow condition ablation")):
            if batch_idx >= args.max_batches:
                break

            audio, pos, neg = batch[:3]
            audio = audio.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            mixture = audio.sum(dim=1).squeeze(1)
            target = audio[:, : conf.active_num[1]].sum(dim=1).squeeze(1)

            conditions = model.make_condition(pos, neg)
            correct_c = conditions["correct"]

            mixture_si = si_snr(mixture, target)
            mixture_l1 = l1_audio(mixture, target)
            mixture_total["si_snr"] += mixture_si.sum().item()
            mixture_total["l1_target"] += mixture_l1.sum().item()

            preds = {}
            for name in variants:
                preds[name] = model.decode_with_condition(mixture, conditions[name])

            correct_pred = preds["correct"]

            for name in variants:
                pred = preds[name]
                score = si_snr(pred, target)
                l1_target = l1_audio(pred, target)
                l1_correct = l1_audio(pred, correct_pred)
                l1_mix = l1_audio(pred, mixture)
                cond_l1 = (conditions[name] - correct_c).abs().mean(dim=(1, 2, 3))

                totals[name]["si_snr"] += score.sum().item()
                totals[name]["si_snri"] += (score - mixture_si).sum().item()
                totals[name]["l1_target"] += l1_target.sum().item()
                totals[name]["l1_correct"] += l1_correct.sum().item()
                totals[name]["l1_mixture"] += l1_mix.sum().item()
                totals[name]["cond_l1"] += cond_l1.sum().item()

                for b in range(target.shape[0]):
                    rows.append(
                        {
                            "batch": batch_idx,
                            "item": b,
                            "variant": name,
                            "mixture_si_snr": float(mixture_si[b].detach().cpu()),
                            "si_snr": float(score[b].detach().cpu()),
                            "si_snri": float((score[b] - mixture_si[b]).detach().cpu()),
                            "l1_target": float(l1_target[b].detach().cpu()),
                            "l1_correct": float(l1_correct[b].detach().cpu()),
                            "l1_mixture": float(l1_mix[b].detach().cpu()),
                            "cond_l1": float(cond_l1[b].detach().cpu()),
                        }
                    )

            if args.save_wavs and saved < args.save_samples:
                for b in range(target.shape[0]):
                    if saved >= args.save_samples:
                        break
                    sample_dir = out_dir / f"sample_{saved:03d}"
                    save_wav(sample_dir / "mixture.wav", mixture[b], conf.sample_rate)
                    save_wav(sample_dir / "target.wav", target[b], conf.sample_rate)
                    for name in variants:
                        save_wav(sample_dir / f"{name}.wav", preds[name][b], conf.sample_rate)
                    saved += 1

            count += target.shape[0]

    print("")
    print("===== RESULTS =====")
    print("examples:", count)
    print(f"mixture_si_snr:    {mixture_total['si_snr'] / max(count, 1):.4f}")
    print(f"mixture_l1_target: {mixture_total['l1_target'] / max(count, 1):.6f}")
    print("")
    print(
        f"{'variant':<20} "
        f"{'si_snr':>10} "
        f"{'si_snri':>10} "
        f"{'l1_target':>12} "
        f"{'l1_correct':>12} "
        f"{'l1_mixture':>12} "
        f"{'cond_l1':>10}"
    )

    for name in variants:
        row = totals[name]
        print(
            f"{name:<20} "
            f"{row['si_snr'] / max(count, 1):>10.4f} "
            f"{row['si_snri'] / max(count, 1):>10.4f} "
            f"{row['l1_target'] / max(count, 1):>12.6f} "
            f"{row['l1_correct'] / max(count, 1):>12.6f} "
            f"{row['l1_mixture'] / max(count, 1):>12.6f} "
            f"{row['cond_l1'] / max(count, 1):>10.6f}"
        )

    metrics_path = out_dir / "rows.tsv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "batch",
                "item",
                "variant",
                "mixture_si_snr",
                "si_snr",
                "si_snri",
                "l1_target",
                "l1_correct",
                "l1_mixture",
                "cond_l1",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    print("")
    print("saved rows to:", metrics_path)
    if args.save_wavs:
        print("saved wavs to:", out_dir)


if __name__ == "__main__":
    main()
