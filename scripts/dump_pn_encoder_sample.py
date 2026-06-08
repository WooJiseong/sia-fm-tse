#!/usr/bin/env python
"""Dump one sample from pretrained PN-TFGridNet extractor.

This bypasses Encoder.extract() and directly calls the wrapped original model:

  condition = encoder(pos, neg)
  state = encoder.model.init_buffers(...)
  out, state = encoder.model(chunk, condition, state)

Outputs:
  mixture.wav
  target.wav
  pn_output.wav
  pos_sum.wav
  neg_sum.wav
  pos_*.wav
  neg_*.wav
  metrics.txt
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import torch
import torchaudio

from sia_fm_tse.model import Encoder
from sia_fm_tse.utils import LibriDataset, load_config


def si_snr(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)

    ref_energy = torch.sum(reference ** 2, dim=-1, keepdim=True).clamp_min(eps)
    projection = torch.sum(estimate * reference, dim=-1, keepdim=True) * reference / ref_energy
    noise = estimate - projection

    ratio = torch.sum(projection ** 2, dim=-1) / torch.sum(noise ** 2, dim=-1).clamp_min(eps)
    return 10.0 * torch.log10(ratio.clamp_min(eps))


def safe_audio(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().cpu()
    peak = x.abs().max().clamp_min(1e-8)
    if peak > 0.99:
        x = x / peak * 0.99
    return x


def save_wav(path: Path, wav: torch.Tensor, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = safe_audio(wav)
    torchaudio.save(str(path), wav.unsqueeze(0), sr)


@torch.no_grad()
def run_original_pn_extractor(
    encoder: Encoder,
    mixture: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run original wrapped PN-TFGridNet model.

    mixture: [B, T]
    pos: [B, N_pos, 1, T]
    neg: [B, N_neg, 1, T]

    returns:
      pn_output: [B, T]
      condition: PN condition tensor
    """
    if mixture.ndim == 2:
        mixture_in = mixture.unsqueeze(1)
    elif mixture.ndim == 3:
        mixture_in = mixture
    else:
        raise ValueError(f"unexpected mixture shape: {tuple(mixture.shape)}")

    if mixture_in.shape[1] != 1:
        raise ValueError(f"expected mixture [B,1,T], got {tuple(mixture_in.shape)}")

    condition = encoder(pos, neg)

    # Original causal PN model uses streaming buffers.
    state = encoder.model.init_buffers(mixture_in.shape[0], mixture_in.device)

    outputs = []
    for chunk in torch.split(mixture_in, chunk_size, dim=-1):
        result = encoder.model(chunk, condition, state)

        # Most likely: (out, state)
        if isinstance(result, tuple):
            if len(result) == 2:
                out, state = result
            else:
                out = result[0]
                state = result[-1]
        else:
            out = result

        outputs.append(out)

    coarse = torch.cat(outputs, dim=-1)

    # Defensive shape handling.
    # Normal expected output: [B, 1, T]
    while coarse.ndim > 2 and coarse.shape[1] == 1:
        coarse = coarse.squeeze(1)

    if coarse.ndim > 2:
        coarse = coarse.reshape(coarse.shape[0], -1)

    return coarse[..., : mixture_in.shape[-1]], condition


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--encoder_ckpt", default="checkpoints/encoder.pt")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--noise_dir", required=True)
    parser.add_argument("--noise_split", default="tt", choices=["tr", "cv", "tt"])
    parser.add_argument("--out_dir", default="outputs/pn_encoder_sample")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--chunk_size", type=int, default=16000)
    args = parser.parse_args()

    conf = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("===== PN Encoder / Original Extractor Sample Dump =====", flush=True)
    print("device:", device, flush=True)
    print("config:", args.config, flush=True)
    print("encoder_ckpt:", args.encoder_ckpt, flush=True)
    print("data_dir:", args.data_dir, flush=True)
    print("noise_dir:", str(Path(args.noise_dir) / args.noise_split) + "/", flush=True)
    print("out_dir:", out_dir, flush=True)
    print("sample_index:", args.sample_index, flush=True)
    print("chunk_size:", args.chunk_size, flush=True)

    encoder = Encoder().to(device)
    encoder.load(Path(args.encoder_ckpt))
    encoder.eval()

    print("wrapped original model:", type(encoder.model), flush=True)

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
        noise_dir=str(Path(args.noise_dir) / args.noise_split) + "/",
        same_disturb=False,
    )

    print("dataset len:", len(dataset), flush=True)

    audio, pos, neg = dataset[args.sample_index]

    audio = audio.unsqueeze(0).to(device)
    pos = pos.unsqueeze(0).to(device)
    neg = neg.unsqueeze(0).to(device)

    print("audio:", tuple(audio.shape), flush=True)
    print("pos:", tuple(pos.shape), flush=True)
    print("neg:", tuple(neg.shape), flush=True)

    mixture = audio.sum(dim=1).squeeze(1)
    target = audio[:, : conf.active_num[1]].sum(dim=1).squeeze(1)

    pn_output, condition = run_original_pn_extractor(
        encoder,
        mixture,
        pos,
        neg,
        chunk_size=args.chunk_size,
    )

    min_len = min(mixture.shape[-1], target.shape[-1], pn_output.shape[-1])
    mixture = mixture[..., :min_len]
    target = target[..., :min_len]
    pn_output = pn_output[..., :min_len]

    mixture_sisnr = si_snr(mixture, target)
    pn_sisnr = si_snr(pn_output, target)

    print("condition:", tuple(condition.shape), flush=True)
    print("pn_output:", tuple(pn_output.shape), flush=True)
    print(f"mixture SI-SNR: {float(mixture_sisnr[0].detach().cpu()):.4f}", flush=True)
    print(f"PN output SI-SNR: {float(pn_sisnr[0].detach().cpu()):.4f}", flush=True)
    print(f"SI-SNR improvement: {float((pn_sisnr - mixture_sisnr)[0].detach().cpu()):.4f}", flush=True)

    save_wav(out_dir / "mixture.wav", mixture[0], conf.sample_rate)
    save_wav(out_dir / "target.wav", target[0], conf.sample_rate)
    save_wav(out_dir / "pn_output.wav", pn_output[0], conf.sample_rate)

    pos_sum = pos[0].sum(dim=0).squeeze(0)
    neg_sum = neg[0].sum(dim=0).squeeze(0)

    save_wav(out_dir / "pos_sum.wav", pos_sum, conf.sample_rate)
    save_wav(out_dir / "neg_sum.wav", neg_sum, conf.sample_rate)

    for i in range(pos.shape[1]):
        save_wav(out_dir / f"pos_{i}.wav", pos[0, i, 0], conf.sample_rate)

    for i in range(neg.shape[1]):
        save_wav(out_dir / f"neg_{i}.wav", neg[0, i, 0], conf.sample_rate)

    with (out_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write(f"sample_index: {args.sample_index}\n")
        f.write(f"mixture_sisnr: {float(mixture_sisnr[0].detach().cpu()):.6f}\n")
        f.write(f"pn_output_sisnr: {float(pn_sisnr[0].detach().cpu()):.6f}\n")
        f.write(f"si_snri: {float((pn_sisnr - mixture_sisnr)[0].detach().cpu()):.6f}\n")
        f.write(f"condition_shape: {tuple(condition.shape)}\n")
        f.write(f"pn_output_shape: {tuple(pn_output.shape)}\n")
        f.write(f"wrapped_model: {type(encoder.model)}\n")

    print("saved wavs to:", out_dir, flush=True)


if __name__ == "__main__":
    main()
