"""
Code for the Encoder Fusion Module
Adopted from the TFGridnet code provided in USEF-TSE: Universal Speaker Embedding Free Target Speaker Extraction
- https://github.com/ZBang/USEF-TSE
  - Original code licensed under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
The modification includes adding the encoding branch, and split trainable parameters between encoding branch and extraction branch
"""

import copy
from collections.abc import Sequence
from typing import Literal

import einops
import torch
import torch.nn as nn

from improved_model.TFgridnet import GridNetV2Block, TFGridNetBlockAttn


class STFT(nn.Module):
    """
    Thin wrapper around torch.stft that supports 2D and 3D inputs and
    returns all commonly needed spectral representations at once.

    Args:
        n_fft:       FFT size.
        hop_length:  Hop size between frames.
        win_length:  Window length.

    Input:
        x: [B, n_samples] or [B, C, n_samples]

    Output (all derived from the same complex STFT):
        magnitude:   |STFT|,        same shape as complex_stft
        phase:       angle(STFT),   same shape as complex_stft
        real:        STFT.real,     same shape as complex_stft
        imag:        STFT.imag,     same shape as complex_stft
        complex_stft: complex tensor [B, F, T] or [B, C, F, T]
    """

    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 128,
        win_length: int = 256,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, x: torch.Tensor):
        # x: [B, n_samples] or [B, C, n_samples]
        n_dims = x.dim()
        assert n_dims in (2, 3), f"Only support 2D or 3D Input: {n_dims}"

        B = x.shape[0]

        # torch.stft only accepts 1D (unbatched) or 2D (batched) input,
        # so merge the channel dim into the batch dim before calling it.
        if n_dims == 3:
            x = einops.rearrange(x, "b c t -> (b c) t")

        # Hann window is recreated on the correct device every forward pass.
        complex_stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=x.device),
            return_complex=True,
        )  # [(B*C), F, T] or [B, F, T]

        # Restore the channel dim that was merged above.
        if n_dims == 3:
            complex_stft = einops.rearrange(complex_stft, "(b c) f t -> b c f t", b=B)

        magnitude = torch.abs(complex_stft)
        phase = torch.angle(complex_stft)
        real = complex_stft.real
        imag = complex_stft.imag
        return magnitude, phase, real, imag, complex_stft


class iSTFT(nn.Module):
    """
    Thin wrapper around torch.istft that accepts three input formats
    and handles complex tensor assembly internally.

    Args:
        n_fft:       FFT size (must match the STFT used to produce the input).
        hop_length:  Hop size.
        win_length:  Window length.
        length:      If given, the output signal is trimmed or zero-padded to
                     exactly this many samples (passed through to torch.istft).

    Input:
        features:   Depends on input_type:
                    - "real_imag":  (real, imag) tuple, each [B, F, T]
                    - "complex":    complex tensor [B, F, T]
                    - "mag_phase":  (magnitude, phase) tuple, each [B, F, T]
        input_type: One of "real_imag", "complex", "mag_phase".

    Output:
        Reconstructed waveform: [B, n_samples]
    """

    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 128,
        win_length: int = 256,
        length: int | None = None,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.length = length

    def forward(
        self,
        features: Sequence[torch.Tensor] | torch.Tensor,
        input_type: Literal["real_imag", "complex", "mag_phase"],
    ) -> torch.Tensor:
        if input_type == "real_imag":
            # the feature is (real, imag)
            assert isinstance(features, Sequence) and len(features) == 2
            real, imag = features
            reconstructed = torch.complex(real, imag)
        elif input_type == "complex":
            assert isinstance(features, torch.Tensor) and torch.is_complex(features), (
                "The input feature is not complex."
            )
            reconstructed = features
        elif input_type == "mag_phase":
            # the feature is (mag, phase)
            assert isinstance(features, Sequence) and len(features) == 2
            magnitude, phase = features
            # Convert polar form to complex: r * e^{j*theta} = r*cos(theta) + j*r*sin(theta)
            reconstructed = torch.complex(
                magnitude * torch.cos(phase),
                magnitude * torch.sin(phase),
            )
        else:
            raise NotImplementedError(
                "Only 'real_imag', 'complex', and 'mag_phase' are supported."
            )

        return torch.istft(
            reconstructed,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=reconstructed.device),
            length=self.length,
        )  # [B, n_samples]


class Tar_Model(nn.Module):
    """
    Target speaker extraction model.

    Architecture overview:
      1. encoder:          STFT + Conv2d to embed the mixture waveform.
      2. attention_block:  Cross-attention to condition the mixture embedding
                           on the speaker embedding (emb).
      3. dual_mdl:         Stack of GridNetV2Blocks (intra/inter RNN + attention)
                           that refine the conditioned representation.
      4. decoder:          ConvTranspose2d + iSTFT to reconstruct the waveform.

    The siamese encoder and encoder_head (conditioning side) can optionally be
    frozen during training via train_encoder / train_encoder_head flags.

    Args:
        n_freqs:            Number of frequency bins (= n_fft // 2 + 1).
        hidden_channels:    Hidden size of the LSTM layers in GridNetV2Block.
        n_head:             Number of attention heads.
        emb_dim:            Channel dimension of the internal embedding.
        emb_ks:             Chunk size for intra/inter LSTM unfolding.
        emb_hs:             Hop size for intra/inter LSTM unfolding.
        num_layers:         Number of GridNetV2Block layers.
        eps:                Epsilon for GroupNorm.
        encoder:            Pre-built siamese encoder module (required if train_encoder).
        encoder_head:       Pre-built encoder head module (required if train_encoder_head).
        train_encoder:      If True, siamese encoder gradients are enabled.
        train_encoder_head: If True, encoder head gradients are enabled.
        binaural:           If True, output two channels (left/right ears).
    """

    def __init__(
        self,
        *,
        n_freqs: int,
        hidden_channels: int,
        n_head: int,
        emb_dim: int,
        emb_ks: int,
        emb_hs: int,
        num_layers: int = 6,
        eps: float = 1e-5,
        encoder: nn.Module,
        encoder_head: nn.Module,
        train_encoder: bool = False,
        train_encoder_head: bool = False,
        binaural: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.binaural = binaural

        # Fixed STFT/iSTFT parameters — must match across encoder and decoder.
        self.stft = STFT(
            n_fft=128,
            hop_length=64,
            win_length=128,
        )
        self.istft = iSTFT(
            n_fft=128,
            hop_length=64,
            win_length=128,
        )

        # Cross-attention block: conditions mixture embedding on speaker embedding.
        self.attention_block = TFGridNetBlockAttn(
            emb_dim=emb_dim,
            n_freqs=n_freqs,
            n_head=4,
            approx_qk_dim=512,
        )

        t_ksize = 3
        kernel_size, padding = (t_ksize, 3), (t_ksize // 2, 1)

        # Input projection: (real, imag) concatenated along channel dim -> emb_dim.
        # binaural uses 4 input channels (2 ears * 2 parts), mono uses 2.
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=4 if binaural else 2,
                out_channels=emb_dim,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )

        # After attention, mixture and conditioned embeddings are concatenated
        # along the channel dim, so the main stream has 2 * emb_dim channels.
        main_emb_dim = 2 * emb_dim

        # Output projection: main_emb_dim -> (real, imag) channels for iSTFT.
        self.deconv = nn.ConvTranspose2d(
            in_channels=main_emb_dim,
            out_channels=4 if binaural else 2,
            kernel_size=kernel_size,
            padding=padding,
        )

        # Deep-copy each block so they have independent weights.
        self.dual_mdl = nn.ModuleList(
            copy.deepcopy(
                GridNetV2Block(
                    emb_dim=main_emb_dim,
                    emb_ks=emb_ks,
                    emb_hs=emb_hs,
                    n_freqs=n_freqs,
                    hidden_channels=hidden_channels,
                    n_head=n_head,
                    approx_qk_dim=512,
                )
            )
            for _ in range(num_layers)
        )

        # Conditioning branch — only registered when training is enabled.
        self.siamese = encoder
        self.encoder_head = encoder_head
        self.train_encoder = train_encoder
        self.train_encoder_head = train_encoder_head

    def to_train(self):
        self.train()

    def encoder_state_dict(self) -> dict:
        """Returns state dicts for both conditioning branch modules."""
        return {
            "siamese": self.siamese.state_dict(),
            "encoder_head": self.encoder_head.state_dict(),
        }

    def encoder_params(self) -> list[nn.Parameter]:
        """Returns parameters belonging to the conditioning branch."""
        modules = [self.siamese, self.encoder_head]
        return [p for m in modules for p in m.parameters()]

    def main_params(self) -> list[nn.Parameter]:
        """Returns parameters belonging to the extraction branch."""
        modules = [self.conv, self.attention_block, self.dual_mdl, self.deconv]
        return [p for m in modules for p in m.parameters()]

    def encoder(self, aux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Embed a mixture waveform into the [B, emb_dim, T, F] representation.

        The input is normalized by its per-sample std before STFT so that the
        model operates on a unit-scale signal; std is returned so the decoder
        can restore the original scale.

        Args:
            aux: [B, 1, n_samples]  (single-channel mixture)

        Returns:
            aux_ri: [B, emb_dim, T, F]  — embedded mixture
            std:    [B, 1, 1]           — per-sample std for scale restoration
        """
        std = aux.std(dim=(1, 2), keepdim=True)  # [B, 1, 1]
        _, _, real, imag, _ = self.stft(aux / std)  # each: [B, C, F, T]

        # Concatenate real and imag along channel dim, then rearrange to [B, 2C, T, F]
        # for Conv2d (which expects spatial dims last).
        aux_ri = torch.cat([real, imag], dim=1)  # [B, 2C, F, T]
        aux_ri = einops.rearrange(aux_ri, "b c f t -> b c t f")  # [B, 2C, T, F]
        aux_ri = self.conv(aux_ri)  # [B, emb_dim, T, F]

        return aux_ri, std

    def encoder_pos_neg(
        self,
        pos: torch.Tensor,
        neg: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        """
        Produce a conditioning embedding from a positive and negative reference pair.

        The siamese encoder maps raw waveforms to embeddings, and the encoder_head
        fuses pos/neg embeddings into a single conditioning signal.
        Gradients for each sub-module are controlled by train_encoder /
        train_encoder_head flags set at construction time.

        Args:
            pos: [B, C, n_samples]  — positive (target speaker) reference
            neg: [B, C, n_samples]  — negative (non-target) reference

        Returns:
            cond_emb: [B, C, T_pos, F]  — conditioning embedding (neg frames discarded)
            None, None                  — placeholder outputs (unused downstream)
        """
        # siamese expects [B, n_samples, C] (sequence-first convention).
        pos = einops.rearrange(pos, "b c n -> b n c")  # n = n_samples
        neg = einops.rearrange(neg, "b c n -> b n c")  # n = n_samples

        if not self.train_encoder:
            with torch.no_grad():
                pos_emb: torch.Tensor = self.siamese(pos).detach()
                neg_emb: torch.Tensor = self.siamese(neg).detach()
        else:
            pos_emb: torch.Tensor = self.siamese(pos)
            neg_emb: torch.Tensor = self.siamese(neg)

        if not self.train_encoder_head:
            with torch.no_grad():
                cond_emb: torch.Tensor = self.encoder_head(pos_emb, neg_emb).detach()
        else:
            cond_emb: torch.Tensor = self.encoder_head(pos_emb, neg_emb)

        # encoder_head may return T_pos + T_neg frames; only T_pos frames are kept
        # because the downstream model expects the conditioning to match the pos length.
        cond_emb = cond_emb[:, :, : pos_emb.shape[2]]  # [B, C, T_pos, F]
        return cond_emb, None, None

    def decoder(self, x: torch.Tensor, std: torch.Tensor):
        """
        Reconstruct a waveform from the refined embedding.

        Channels 0/1 carry (real, imag) for the first ear (or mono);
        channels 2/3 carry (real, imag) for the second ear (binaural only).
        The output is rescaled by std to restore the original signal level.

        Args:
            x:   [B, main_emb_dim, T, F]
            std: [B, 1, 1]

        Returns:
            mono:     [B, n_samples]           (binaural=False)
            binaural: [B, 2, 1, n_samples]     (binaural=True)
        """
        x: torch.Tensor = self.deconv(x)  # [B, 4 or 2, T', F']

        # Rearrange F and T back to the order expected by iSTFT: [B, F, T].
        out_r = einops.rearrange(x[:, 0, :, :], "b t f -> b f t")
        out_i = einops.rearrange(x[:, 1, :, :], "b t f -> b f t")

        est_source: torch.Tensor = self.istft(
            (out_r, out_i),
            input_type="real_imag",
        )  # [B, n_samples]

        # unsqueeze to [B, 1, n_samples] so std [B, 1, 1] broadcasts correctly.
        est_source = einops.rearrange(est_source, "b n -> b 1 n")
        est_source *= std  # restore original scale

        if self.binaural:
            out2_r = einops.rearrange(x[:, 2, :, :], "b t f -> b f t")
            out2_i = einops.rearrange(x[:, 3, :, :], "b t f -> b f t")
            est_source2: torch.Tensor = self.istft(
                (out2_r, out2_i),
                input_type="real_imag",
            )  # [B, n_samples]

            est_source2 = einops.rearrange(est_source2, "b n -> b 1 n")
            est_source2 *= std

            # Stack left and right ear outputs along a new ear dim.
            est_source = torch.stack([est_source, est_source2], dim=1)
            return est_source  # [B, 2, 1, n_samples]
        else:
            return einops.rearrange(est_source, "b 1 n -> b n")  # [B, n_samples]

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: encode -> condition -> refine -> decode.

        Args:
            x:   [B, 1, n_samples]   — mixture waveform
            emb: [B, C, T_emb, F]   — speaker conditioning embedding

        Returns:
            Extracted target speaker waveform (shape depends on binaural flag).
        """
        # Embed the mixture and normalize it.
        mix_ri, std = self.encoder(x)  # [B, emb_dim, T, F], [B, 1, 1]

        # Cross-attend to the speaker embedding.
        aux_ri = self.attention_block(mix_ri, emb)  # [B, emb_dim, T, F]

        # Concatenate original and conditioned embeddings for the main stack.
        x = torch.cat([mix_ri, aux_ri], dim=1)  # [B, 2*emb_dim, T, F]

        # Iterative refinement through GridNet blocks.
        for module in self.dual_mdl:
            x = module(x)

        return self.decoder(x, std)
