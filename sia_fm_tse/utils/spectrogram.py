"""Spectrogram utilities."""

import torch
from einops import rearrange


def stft_torch(
    signal: torch.Tensor,
    *,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int = 512,
) -> torch.Tensor:
    if signal.dim() == 3:
        signal = rearrange(signal, "b 1 t -> b t")

    assert signal.dim() == 2

    window = torch.hann_window(win_length, device=signal.device)
    spec = torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    spec = torch.stack((spec.real, spec.imag), dim=1)
    return rearrange(spec, "b c f n -> b (c f) n")


def istft_torch(
    spec_concat: torch.Tensor,
    *,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int = 512,
    length: int | None = None,
) -> torch.Tensor:
    freq = n_fft // 2 + 1
    spec_concat = rearrange(spec_concat, "b (c f) n -> b c f n", c=2, f=freq)
    spec_real, spec_imag = spec_concat.unbind(dim=1)
    spec = torch.complex(spec_real, spec_imag)

    window = torch.hann_window(win_length, device=spec.device)
    return torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=length,
    )
