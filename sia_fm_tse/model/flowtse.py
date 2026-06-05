import torch
import torch.nn as nn
import torchaudio
from einops import rearrange
from vocos import Vocos

from ..utils import get_vocos_mel_spectrogram
from .decoder import CFM as Decoder
from .encoder import Encoder


class FlowTSE(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        """
        Full model for TSE with PN encoder and CFM-based decoder.
        Condition c is injected via cross-attention inside DiT.

         pos --> ┌─────────┐
                 │ Encoder │ --> c  (reduce + interpolate to mel T)
         neg --> └─────────┘

         x  --> resample --> mel --> ┌─────────┐
                                     │ Decoder │ --> waveform (w/ trajectory)
         c ------------------------> └─────────┘
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(
            self.decoder.device
        )
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=16000, new_freq=24000
        ).to(self.decoder.device)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
        *,
        steps: int = 32,
        cfg_strength: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample clean audio from mixture.

        Args:
            x: [B, T_audio] mixture waveform at 16kHz
            pos: [B, n_pos, T_audio] positive enrollment waveforms at 16kHz
            neg: [B, n_neg, T_audio] negative enrollment waveforms at 16kHz

        Returns:
            out:        [B, nw] output waveform
            trajectory: [steps+1, B, T, mel_dim] ODE trajectory
        """
        c = self.encoder(pos, neg)  # [B, C, T_enc, F] — passed directly

        x = self.resampler(x)
        m = get_vocos_mel_spectrogram(
            x, n_mel_channels=self.decoder.transformer.mel_dim
        )
        m = rearrange(m, "b d n -> b n d")  # [B, T_mel, mel_dim]

        out, trajectory = self.decoder(
            m,
            c,
            steps=steps,
            cfg_strength=cfg_strength,
            vocoder=self.vocoder,
        )
        return out, trajectory
