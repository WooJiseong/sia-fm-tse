import torch
import torch.nn as nn
import torchaudio
from einops import reduce
from vocos import Vocos

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

         pos --> ┌─────────┐
                 │ Encoder │ --> c
         neg --> └─────────┘

         m(=x) -->  ┌─────────┐
                    │ Decoder │ --> output waveform/mel-spectrogram
         c -------> └─────────┘      (w/ trajectory)

        On traning, you should use
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
        Sample clean audio from mixture

        Args:
            x: [B, C, T, mel_dim] audio mixture
            pos: [B, C, T, mel_dim] positive embedding
            neg: [B, C, T, mel_dim] positive embedding

        Returns:
            out: [B, nw]
            trajectory: [steps+1, B, T, mel_dim] ODE trajectory
        """

        c = self.encoder(pos, neg)
        c = reduce(c, "b c t f -> b t f", "mean")
        x = self.resampler(x)

        out, trajectory = self.decoder(
            x,
            c,
            steps=steps,
            cfg_strength=cfg_strength,
            vocoder=self.vocoder,
        )
        return out, trajectory
