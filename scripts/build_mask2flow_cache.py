#!/usr/bin/env python
"""Build cached STFT triples for Mask2Flow Phase A experiments.

Cache item:
  mixture_stft: [T, D]
  coarse_stft:  [T, D]
  target_stft:  [T, D]
  condition:    PN encoder condition
  length:       waveform length
  coarse_mode:  mixture | oracle_irm | oracle_complex
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from sia_fm_tse.model import Encoder
from sia_fm_tse.utils import LibriDataset, load_config, stft_torch


def get_conf(conf: Any, key: str, default: Any = None) -> Any:
    """Support both dict-like and attribute-like configs."""
    if isinstance(conf, dict):
        return conf.get(key, default)
    return getattr(conf, key, default)


def _concat_to_complex(spec: torch.Tensor, n_fft: int) -> torch.Tensor:
    """Convert concat real/imag STFT [B, 2F, T] -> complex [B, F, T]."""
    freq = n_fft // 2 + 1

    if spec.dim() != 3:
        raise ValueError(f"expected STFT tensor [B, 2F, T], got {tuple(spec.shape)}")

    if spec.shape[1] != 2 * freq:
        raise ValueError(
            f"STFT dim mismatch: expected 2F={2 * freq}, got {spec.shape[1]}"
        )

    spec = rearrange(spec, "b (c f) t -> b c f t", c=2, f=freq)
    return torch.complex(spec[:, 0], spec[:, 1])


def _complex_to_concat(spec: torch.Tensor) -> torch.Tensor:
    """Convert complex STFT [B, F, T] -> concat real/imag [B, 2F, T]."""
    spec = torch.stack((spec.real, spec.imag), dim=1)
    return rearrange(spec, "b c f t -> b (c f) t")


def _make_coarse(
    mode: str,
    mixture_stft: torch.Tensor,
    target_stft: torch.Tensor,
    n_fft: int,
) -> torch.Tensor:
    """Build S0 coarse target estimate."""
    if mode == "mixture":
        return mixture_stft

    if mode == "oracle_complex":
        return target_stft

    if mode == "oracle_irm":
        mixture = _concat_to_complex(mixture_stft, n_fft)
        target = _concat_to_complex(target_stft, n_fft)

        mask = (target.abs() / (mixture.abs() + 1e-8)).clamp(0.0, 1.0)
        coarse = mask * mixture

        return _complex_to_concat(coarse)

    raise ValueError("coarse_mode must be one of: mixture, oracle_irm, oracle_complex")


def _summarize_tensor(name: str, x: torch.Tensor) -> None:
    print(
        f"{name}: shape={tuple(x.shape)}, dtype={x.dtype}, "
        f"min={float(x.min().detach().cpu()):.6f}, "
        f"max={float(x.max().detach().cpu()):.6f}",
        flush=True,
    )


def main() -> None:
    parser = ArgumentParser(description="Cache S0/Y/S/PN-condition tensors.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--noise_dir", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument(
        "--coarse_mode",
        default="oracle_irm",
        choices=["mixture", "oracle_irm", "oracle_complex"],
    )
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    conf = load_config(args.config)

    sample_rate = int(get_conf(conf, "sample_rate"))
    n_fft = int(get_conf(conf, "n_fft"))
    hop_length = int(get_conf(conf, "hop_length"))
    win_length = int(get_conf(conf, "win_length"))
    snr_db_range = get_conf(conf, "snr_db_range")
    min_source_num = get_conf(conf, "min_source_num")
    source_num = get_conf(conf, "source_num")
    active_num = get_conf(conf, "active_num")
    encoder_ckpt_path = get_conf(conf, "encoder_ckpt_path")

    if encoder_ckpt_path is None:
        raise RuntimeError("encoder_ckpt_path is missing in config")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print("===== Mask2Flow cache builder =====", flush=True)
    print("device:", device, flush=True)
    print("config:", args.config, flush=True)
    print("data_dir:", args.data_dir, flush=True)
    print("noise_dir:", args.noise_dir, flush=True)
    print("cache_dir:", args.cache_dir, flush=True)
    print("coarse_mode:", args.coarse_mode, flush=True)
    print("num_samples:", args.num_samples, flush=True)
    print("batch_size:", args.batch_size, flush=True)
    print("sample_rate:", sample_rate, flush=True)
    print("n_fft:", n_fft, flush=True)
    print("hop_length:", hop_length, flush=True)
    print("win_length:", win_length, flush=True)
    print("active_num:", active_num, flush=True)
    print("encoder_ckpt_path:", encoder_ckpt_path, flush=True)

    if not args.data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {args.data_dir}")

    noise_tr_dir = args.noise_dir / "tr"
    if not noise_tr_dir.exists():
        raise FileNotFoundError(f"noise tr dir not found: {noise_tr_dir}")

    encoder = Encoder().to(device)
    encoder.load(encoder_ckpt_path)
    encoder.eval()

    dataset = LibriDataset(
        str(args.data_dir),
        sample_rate=sample_rate,
        wave_length=3 * sample_rate,
        pos_example_length=3 * sample_rate,
        neg_example_length=3 * sample_rate,
        snr_db_range=snr_db_range,
        min_source_num=min_source_num,
        source_num=source_num,
        active_num=active_num,
        reproducable=False,
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
        noise_dir=str(noise_tr_dir) + "/",
        same_disturb=False,
    )

    if len(dataset) <= 0:
        raise RuntimeError("LibriDataset length is 0")

    print("dataset len:", len(dataset), flush=True)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    if len(loader) <= 0:
        raise RuntimeError("DataLoader length is 0")

    print("loader len:", len(loader), flush=True)

    saved = 0
    first_batch_printed = False

    with torch.no_grad():
        while saved < args.num_samples:
            made_progress = False

            for batch in tqdm(loader, desc="cache", dynamic_ncols=True):
                if not isinstance(batch, (tuple, list)) or len(batch) < 3:
                    raise RuntimeError(
                        f"expected batch=(audio,pos,neg), got type={type(batch)}, len={len(batch) if hasattr(batch, '__len__') else 'NA'}"
                    )

                audio, pos, neg = batch[:3]

                audio = audio.to(device)
                pos = pos.to(device)
                neg = neg.to(device)

                if not first_batch_printed:
                    print("===== first batch =====", flush=True)
                    _summarize_tensor("audio", audio)
                    _summarize_tensor("pos", pos)
                    _summarize_tensor("neg", neg)
                    first_batch_printed = True

                # audio expected: [B, source_num, 1, T] or similar
                if audio.dim() == 4:
                    mixture = audio.sum(dim=1).squeeze(1)
                    target = audio[:, : int(active_num[1])].sum(dim=1).squeeze(1)
                elif audio.dim() == 3:
                    mixture = audio.sum(dim=1)
                    target = audio[:, : int(active_num[1])].sum(dim=1)
                else:
                    raise RuntimeError(f"unexpected audio shape: {tuple(audio.shape)}")

                condition = encoder(pos, neg)

                mixture_stft = stft_torch(
                    mixture,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                )
                target_stft = stft_torch(
                    target,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                )
                coarse_stft = _make_coarse(
                    args.coarse_mode,
                    mixture_stft,
                    target_stft,
                    n_fft,
                )

                if not made_progress:
                    print("===== first computed tensors =====", flush=True)
                    _summarize_tensor("mixture", mixture)
                    _summarize_tensor("target", target)
                    _summarize_tensor("condition", condition)
                    _summarize_tensor("mixture_stft", mixture_stft)
                    _summarize_tensor("target_stft", target_stft)
                    _summarize_tensor("coarse_stft", coarse_stft)

                for idx in range(mixture.shape[0]):
                    if saved >= args.num_samples:
                        break

                    item = {
                        "mixture_stft": rearrange(mixture_stft[idx].detach().cpu(), "d t -> t d"),
                        "coarse_stft": rearrange(coarse_stft[idx].detach().cpu(), "d t -> t d"),
                        "target_stft": rearrange(target_stft[idx].detach().cpu(), "d t -> t d"),
                        "condition": condition[idx].detach().cpu(),
                        "length": int(mixture.shape[-1]),
                        "coarse_mode": args.coarse_mode,
                    }

                    out_path = args.cache_dir / f"sample_{saved:07d}.pt"
                    torch.save(item, out_path)

                    saved += 1
                    made_progress = True

                    if saved <= 3 or saved == args.num_samples:
                        print(f"saved {saved}/{args.num_samples}: {out_path}", flush=True)

                if saved >= args.num_samples:
                    break

            if not made_progress:
                raise RuntimeError(
                    "No samples were saved in one full DataLoader pass. Check dataset/batch shapes."
                )

    print(f"saved {saved} samples to {args.cache_dir}", flush=True)


if __name__ == "__main__":
    main()
