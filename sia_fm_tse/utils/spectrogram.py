"""Modified from https://github.com/xu-shitong/TSE-through-Positive-Negative-Enroll"""

import torch
import torchaudio


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
