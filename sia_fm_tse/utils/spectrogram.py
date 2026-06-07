"""Spectrogram utilities."""

import torch
import torchaudio
from einops import rearrange


def get_vocos_mel_spectrogram(
    waveform: torch.Tensor,
    n_fft: int = 1024,
    n_mel_channels: int = 100,
    target_sample_rate: int = 24000,
    hop_length: int = 256,
    win_length: int = 1024,
):
    mel_stft = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mel_channels,
        power=1,
        center=True,
        normalized=False,
        norm=None,
    ).to(waveform.device)
    if len(waveform.shape) == 3:
        waveform = waveform.squeeze(1)  # 'b 1 nw -> b nw'

    assert len(waveform.shape) == 2

    mel = mel_stft(waveform)
    mel = mel.clamp(min=1e-5).log()
    return mel


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
