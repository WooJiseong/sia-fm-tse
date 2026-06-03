import torch
import torch.nn as nn

from .decoder import CFM as Decoder
from .encoder import Encoder


class FlowTSE(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        """
        full model for TSE PN encoder and CFM based Decoder

        **WARNING**
        This model uses just condition concatenation,
        not cross-attention based injection.
        > See decoder.transformer.DiT

        Sampling:

         x ----> ┌─────────┐
         pos --> │ Encoder │ --> c
         neg --> └─────────┘

         m(=x) -->  ┌─────────┐
                    │ Decoder │ --> output waveform/mel-spectrogram
         c -------> └─────────┘      (w/ trajectory)

        On traning, you should use
        """
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
        *,
        steps: int = 32,
        cfg_strength: float = 1.0,
        vocoder: nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample clean audio from mixture

        Args:
            x: [B, T, mel_dim] audio mixture
            pos: [B, T, mel_dim] positive embedding
            neg: [B, T, mel_dim] positive embedding

        Returns:
            out: [B, T, mel_dim] or [B, nw] if vocoder is provided
            trajectory: [steps+1, B, T, mel_dim] ODE trajectory
        """

        c = self.encoder(x, pos, neg)
        out, trajectory = self.decoder(
            x,
            c,
            steps=steps,
            cfg_strength=cfg_strength,
            vocoder=vocoder,
        )
        return out, trajectory
