#!/usr/bin/env python
"""Evaluate Mask2Flow refiner using CAv2 utils/eval.py.

This wrapper adapts Mask2FlowCFM to the interface expected by sia_fm_tse.utils.eval:

    pred, _ = model(mixture, pos, neg)

This script is intended for the mixture-start Mask2Flow checkpoint:

    S0 = mixture STFT
    refined = Mask2Flow(S0, mixture STFT, PN condition)

Do NOT use this for oracle_irm checkpoint in fair evaluation, because oracle_irm
requires clean target to construct S0.
"""

from __future__ import annotations

import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange

from sia_fm_tse.model import Encoder, Mask2FlowCFM, Mask2FlowDiT
from sia_fm_tse.utils import istft_torch, load_eval_config, stft_torch
from sia_fm_tse.utils.eval import eval as cav2_eval


class Mask2FlowEvalWrapper(nn.Module):
    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        encoder_ckpt_path: str | Path,
        n_fft: int,
        hop_length: int,
        win_length: int,
        steps: int = 8,
        cfg_strength: float = 0.0,
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
        ckpt_conf: dict[str, Any] = ckpt["config"]
        model_conf = ckpt_conf["model"]

        stft_dim = 2 * (int(model_conf["n_fft"]) // 2 + 1)

        if stft_dim != 2 * (self.n_fft // 2 + 1):
            raise ValueError(
                f"STFT dim mismatch: checkpoint stft_dim={stft_dim}, "
                f"eval config stft_dim={2 * (self.n_fft // 2 + 1)}"
            )

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

        print("===== Mask2FlowEvalWrapper loaded =====", flush=True)
        print("checkpoint:", checkpoint_path, flush=True)
        print("encoder_ckpt:", encoder_ckpt_path, flush=True)
        print("steps:", self.steps, flush=True)
        print("cfg_strength:", self.cfg_strength, flush=True)
        print("n_fft:", self.n_fft, flush=True)
        print("hop_length:", self.hop_length, flush=True)
        print("win_length:", self.win_length, flush=True)

    @torch.no_grad()
    def forward(
        self,
        mixture: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """Return waveform prediction for CAv2 eval.py.

        Args:
            mixture: [B, T]
            pos: [B, N_pos, 1, T]
            neg: [B, N_neg, 1, T]

        Returns:
            pred_wave: [B, T]
            None
        """
        length = mixture.shape[-1]

        condition = self.encoder(pos, neg)

        mixture_stft = stft_torch(
            mixture,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        mixture_stft = rearrange(mixture_stft, "b d t -> b t d")

        # mixture-start evaluation:
        # S0 = mixture STFT
        start = mixture_stft

        refined_stft, _ = self.refiner(
            start=start,
            mixture=mixture_stft,
            c=condition,
            steps=self.steps,
            cfg_strength=self.cfg_strength,
        )

        refined_stft = rearrange(refined_stft, "b t d -> b d t")

        pred_wave = istft_torch(
            refined_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            length=length,
        )

        return pred_wave, None


def main() -> None:
    parser = ArgumentParser(description="Evaluate Mask2Flow with CAv2 utils/eval.py")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--noise_dir", required=True)
    parser.add_argument("--noise_split", default="tt", choices=["tr", "cv", "tt"])
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg_strength", type=float, default=0.0)
    parser.add_argument("--log_file", default="outputs/mask2flow_eval/mixture_cav2_eval.log")
    args = parser.parse_args()

    conf = load_eval_config(args.config)

    noise_path = Path(args.noise_dir)
    if (noise_path / args.noise_split).exists():
        noise_path = noise_path / args.noise_split

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("mask2flow_cav2_eval")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)

    model = Mask2FlowEvalWrapper(
        checkpoint_path=args.checkpoint,
        encoder_ckpt_path=conf.encoder_ckpt_path,
        n_fft=conf.n_fft,
        hop_length=conf.hop_length,
        win_length=conf.win_length,
        steps=args.steps,
        cfg_strength=args.cfg_strength,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print("===== RUN CAv2 eval.py =====", flush=True)
    print("config:", args.config, flush=True)
    print("checkpoint:", args.checkpoint, flush=True)
    print("data_dir:", args.data_dir, flush=True)
    print("noise_dir:", str(noise_path) + "/", flush=True)
    print("log_file:", log_path, flush=True)

    results = cav2_eval(
        model,
        conf,
        data_dir=args.data_dir,
        noise_dir=str(noise_path) + "/",
        handler=handler,
    )

    print("===== RESULTS =====", flush=True)
    for k, v in results.items():
        print(f"{k}: {v:.6f}", flush=True)

    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n===== FINAL RESULTS =====\n")
        for k, v in results.items():
            f.write(f"{k}: {v:.6f}\n")


if __name__ == "__main__":
    main()
