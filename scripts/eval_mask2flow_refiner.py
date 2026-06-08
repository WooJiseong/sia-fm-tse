#!/usr/bin/env python
"""Evaluate/listen to Mask2Flow Phase A refiner from cached STFT samples."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
import torchaudio
import yaml
from einops import rearrange
from tqdm import tqdm

from sia_fm_tse.model import Mask2FlowCFM, Mask2FlowDiT


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _concat_to_complex(spec: torch.Tensor, n_fft: int) -> torch.Tensor:
    """[B, T, 2F] -> [B, F, T] complex."""
    freq = n_fft // 2 + 1
    if spec.dim() != 3:
        raise ValueError(f"expected [B,T,2F], got {tuple(spec.shape)}")
    if spec.shape[-1] != 2 * freq:
        raise ValueError(f"expected last dim {2 * freq}, got {spec.shape[-1]}")

    spec = rearrange(spec, "b t (c f) -> b c f t", c=2, f=freq)
    return torch.complex(spec[:, 0], spec[:, 1])


def _stft_to_wav(
    spec: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    length: int,
) -> torch.Tensor:
    """Convert concat real/imag STFT [B,T,2F] to waveform [B,T_audio]."""
    complex_spec = _concat_to_complex(spec, n_fft)
    window = torch.hann_window(win_length, device=spec.device, dtype=spec.dtype)

    wav = torch.istft(
        complex_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=length,
    )
    return wav


def _safe_audio(x: torch.Tensor) -> torch.Tensor:
    """Normalize only if clipping would occur."""
    x = x.detach().cpu()
    peak = x.abs().max().clamp_min(1e-8)
    if peak > 0.99:
        x = x / peak * 0.99
    return x


def _save_wav(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = _safe_audio(wav)
    torchaudio.save(str(path), wav.unsqueeze(0), sample_rate)


def _load_model(ckpt_path: Path, device: torch.device) -> tuple[Mask2FlowCFM, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    conf = ckpt["config"]

    model_conf = conf["model"]
    n_fft = int(model_conf["n_fft"])
    stft_dim = 2 * (n_fft // 2 + 1)

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
        cond_drop_prob=float(model_conf.get("cond_drop_prob", 0.0)),
    )

    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    return model, conf


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--out_dir", default="outputs/mask2flow_eval/oracle_irm")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg_strength", type=float, default=0.0)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--hop_length", type=int, default=128)
    parser.add_argument("--win_length", type=int, default=512)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    model, conf = _load_model(ckpt_path, device)

    if args.cache_dir is None:
        cache_dir = Path(conf["paths"]["cache_dir"])
    else:
        cache_dir = Path(args.cache_dir)

    files = sorted(cache_dir.glob("sample_*.pt"))
    if not files:
        raise RuntimeError(f"no cache files found in {cache_dir}")

    files = files[: args.num_samples]

    print("device:", device, flush=True)
    print("ckpt:", ckpt_path, flush=True)
    print("cache_dir:", cache_dir, flush=True)
    print("out_dir:", out_dir, flush=True)
    print("num_samples:", len(files), flush=True)
    print("steps:", args.steps, flush=True)
    print("cfg_strength:", args.cfg_strength, flush=True)

    metrics = []

    with torch.no_grad():
        for i, path in enumerate(tqdm(files, desc="eval")):
            item = torch.load(path, map_location="cpu")

            mixture = item["mixture_stft"].unsqueeze(0).to(device)
            coarse = item["coarse_stft"].unsqueeze(0).to(device)
            target = item["target_stft"].unsqueeze(0).to(device)
            condition = item["condition"].unsqueeze(0).to(device)
            length = int(item.get("length", 48000))

            refined, _trajectory = model(
                start=coarse,
                mixture=mixture,
                c=condition,
                steps=args.steps,
                cfg_strength=args.cfg_strength,
            )

            coarse_l1 = torch.nn.functional.l1_loss(coarse, target).item()
            refined_l1 = torch.nn.functional.l1_loss(refined, target).item()
            coarse_mse = torch.nn.functional.mse_loss(coarse, target).item()
            refined_mse = torch.nn.functional.mse_loss(refined, target).item()

            metrics.append(
                {
                    "idx": i,
                    "file": path.name,
                    "coarse_l1": coarse_l1,
                    "refined_l1": refined_l1,
                    "coarse_mse": coarse_mse,
                    "refined_mse": refined_mse,
                    "l1_improvement": coarse_l1 - refined_l1,
                    "mse_improvement": coarse_mse - refined_mse,
                }
            )

            mixture_wav = _stft_to_wav(
                mixture,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                win_length=args.win_length,
                length=length,
            )[0]
            coarse_wav = _stft_to_wav(
                coarse,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                win_length=args.win_length,
                length=length,
            )[0]
            refined_wav = _stft_to_wav(
                refined,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                win_length=args.win_length,
                length=length,
            )[0]
            target_wav = _stft_to_wav(
                target,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                win_length=args.win_length,
                length=length,
            )[0]

            sample_dir = out_dir / f"sample_{i:03d}"
            _save_wav(sample_dir / "mixture.wav", mixture_wav, args.sample_rate)
            _save_wav(sample_dir / "coarse_oracle_irm.wav", coarse_wav, args.sample_rate)
            _save_wav(sample_dir / "refined.wav", refined_wav, args.sample_rate)
            _save_wav(sample_dir / "target.wav", target_wav, args.sample_rate)

    if metrics:
        mean_coarse_l1 = sum(m["coarse_l1"] for m in metrics) / len(metrics)
        mean_refined_l1 = sum(m["refined_l1"] for m in metrics) / len(metrics)
        mean_coarse_mse = sum(m["coarse_mse"] for m in metrics) / len(metrics)
        mean_refined_mse = sum(m["refined_mse"] for m in metrics) / len(metrics)

        print("===== METRICS =====", flush=True)
        print(f"mean coarse L1:  {mean_coarse_l1:.6f}", flush=True)
        print(f"mean refined L1: {mean_refined_l1:.6f}", flush=True)
        print(f"L1 improvement:  {mean_coarse_l1 - mean_refined_l1:.6f}", flush=True)
        print(f"mean coarse MSE:  {mean_coarse_mse:.6f}", flush=True)
        print(f"mean refined MSE: {mean_refined_mse:.6f}", flush=True)
        print(f"MSE improvement:  {mean_coarse_mse - mean_refined_mse:.6f}", flush=True)

    metric_path = out_dir / "metrics.tsv"
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    with metric_path.open("w", encoding="utf-8") as f:
        f.write("idx\tfile\tcoarse_l1\trefined_l1\tcoarse_mse\trefined_mse\tl1_improvement\tmse_improvement\n")
        for m in metrics:
            f.write(
                f"{m['idx']}\t{m['file']}\t"
                f"{m['coarse_l1']:.8f}\t{m['refined_l1']:.8f}\t"
                f"{m['coarse_mse']:.8f}\t{m['refined_mse']:.8f}\t"
                f"{m['l1_improvement']:.8f}\t{m['mse_improvement']:.8f}\n"
            )

    print("saved wavs to:", out_dir, flush=True)
    print("saved metrics to:", metric_path, flush=True)


if __name__ == "__main__":
    main()
