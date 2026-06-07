import torch
import torch.nn as nn
from einops import rearrange

from ..utils import istft_torch, stft_torch
from .decoder import CFM as Decoder
from .encoder import Encoder


class FlowTSE(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        *,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
    ):
        """
        Full model for TSE with PN encoder and CFM-based decoder.
        Condition c is injected via cross-attention inside DiT.

         pos --> ┌─────────┐
                 │ Encoder │ --> c  (reduce + interpolate to mel T)
         neg --> └─────────┘

         x  --> STFT --> ┌─────────┐
                         │ Decoder │ --> ISTFT waveform (w/ trajectory)
         c ------------> └─────────┘
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

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
            trajectory: [steps+1, B, T, stft_dim] ODE trajectory
        """
        c = self.encoder(pos, neg)  # [B, C, T_enc, F] — passed directly

        original_len = x.shape[-1]
        m = stft_torch(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        m = rearrange(m, "b d n -> b n d")  # [B, T_stft, stft_dim]

        out, trajectory = self.decoder(
            m,
            c,
            steps=steps,
            cfg_strength=cfg_strength,
        )
        out = rearrange(out, "b n d -> b d n")
        out = istft_torch(
            out,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            length=original_len,
        )
        return out, trajectory
