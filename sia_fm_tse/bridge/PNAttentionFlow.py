from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from random import random
from typing import Any, TypeVar, cast

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint
from x_transformers.x_transformers import RotaryEmbedding

from sia_fm_tse.flowse.model.backbones.dit import (
    AdaLayerNormZero_Final,
    ConvPositionEmbedding,
    DiTBlock,
    TextEmbedding,
    TimestepEmbedding,
)
from sia_fm_tse.flowse.model.modules import MelSpec
from sia_fm_tse.pnenroll.improved_model.GridnetAttnHead import GridNetBlockAttnHead
from sia_fm_tse.pnenroll.improved_model.TFgridnet import TFGridNetBlockAttn
from sia_fm_tse.pnenroll.improved_model.USEF_TFGridnet import STFT
from sia_fm_tse.pnenroll.model.tfgridnet_encoder import TFGridNet_encoder

# ─────────────────────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────────────────────
# Original `flowse.model.model_utils` module is *typically* crap

_T = TypeVar("_T")


def default(v: _T, d: Any) -> _T:
    return v if v is not None else cast(_T, d)


def list_str_to_tensor(text: list[str], padding_value: int = -1) -> torch.Tensor:
    list_tensors = [torch.tensor([*bytes(t, "UTF-8")]) for t in text]  # ByT5 style
    text_tensor = pad_sequence(
        list_tensors,
        padding_value=padding_value,
        batch_first=True,
    )
    return text_tensor


def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],
    padding_value: int = -1,
) -> torch.Tensor:
    list_idx_tensors = [
        torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text
    ]
    text_tensor = pad_sequence(
        list_idx_tensors,
        padding_value=padding_value,
        batch_first=True,
    )
    return text_tensor


# ─────────────────────────────────────────────────────────────────────────────
# InputEmbedding
# ─────────────────────────────────────────────────────────────────────────────


class InputEmbedding(nn.Module):
    """
    Projects the concatenation of (noisy input, conditioning audio, speaker
    embedding, text embedding) into the DiT model dimension.

    The speaker_embed slot was added on top of the original FlowSE definition,
    which only concatenated (x, cond, text_embed).  That is why the input
    projection expects mel_dim * 3 + text_dim rather than mel_dim * 2 + text_dim.

    Args:
        mel_dim:  Frequency dimension F shared by x, cond, and speaker_embed.
        text_dim: Feature dimension of the text embedding.
        out_dim:  Output model dimension (DiT dim).

    Input:
        x:               [B, T, F]        — noisy/target mel frames
        cond:            [B, T, F]        — conditioning (noisy) mel frames
        speaker_embed:   [B, T, F]        — speaker embedding from PNAttentionFlow
        text_embed:      [B, T, text_dim] — positional text features
        drop_audio_cond: if True, cond is zeroed out (CFG training)

    Output:
        [B, T, out_dim]
    """

    def __init__(self, mel_dim: int, text_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(mel_dim * 3 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(
        self,
        x: torch.Tensor,  # [B, T, F]
        cond: torch.Tensor,  # [B, T, F]
        speaker_embed: torch.Tensor,  # [B, T, F]
        text_embed: torch.Tensor,  # [B, T, text_dim]
        drop_audio_cond: bool = False,
    ) -> torch.Tensor:  # [B, T, out_dim]
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
        x = self.proj(torch.cat((x, cond, speaker_embed, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


# ─────────────────────────────────────────────────────────────────────────────
# DiT
# ─────────────────────────────────────────────────────────────────────────────


class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) conditioned on time, text, and speaker identity.

    This class extends the original FlowSE DiT by threading speaker_embed
    through InputEmbedding so that every transformer block is implicitly
    conditioned on the target speaker.

    Args:
        dim:                    Model dimension.
        depth:                  Number of DiTBlock layers.
        heads:                  Number of attention heads.
        dim_head:               Dimension per attention head.
        dropout:                Dropout rate inside DiTBlock.
        ff_mult:                Feed-forward expansion multiplier.
        mel_dim:                Frequency dimension F of the mel spectrogram.
        text_num_embeds:        Vocabulary size for the text embedding table.
        text_dim:               Feature dimension of text embeddings
                                (defaults to mel_dim if not provided).
        conv_layers:            Number of convolutional layers in TextEmbedding.
        long_skip_connection:   If True, add a linear skip from input to output.
        checkpoint_activations: If True, use gradient checkpointing per block.

    Input (forward):
        x:               [B, T, F]    — noisy mel at timestep t
        cond:            [B, T, F]    — conditioning (noisy) mel
        speaker_embed:   [B, T, F]    — speaker embedding
        text:            [B, N]       — token indices
        time:            [B] or scalar
        drop_audio_cond: bool         — zero out cond for CFG
        drop_text:       bool         — zero out text for CFG
        mask:            [B, T] | None

    Output:
        [B, T, F]  — predicted flow vector
    """

    def __init__(
        self,
        *,
        dim: int,
        depth: int = 8,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1,
        ff_mult: int = 4,
        mel_dim: int = 100,
        text_num_embeds: int = 256,
        text_dim: int | None = None,
        conv_layers: int = 0,
        long_skip_connection: bool = False,
        checkpoint_activations: bool = False,
    ) -> None:
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(
            text_num_embeds, text_dim, conv_layers=conv_layers
        )
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = (
            nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        )
        self.norm_out = AdaLayerNormZero_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)
        self.checkpoint_activations = checkpoint_activations

    def ckpt_wrapper(self, module: nn.Module) -> Callable[..., torch.Tensor]:
        def ckpt_forward(*inputs: Any) -> torch.Tensor:
            return module(*inputs)  # type: ignore[return-value]

        return ckpt_forward

    def forward(
        self,
        x: torch.Tensor,  # [B, T, F]
        cond: torch.Tensor,  # [B, T, F]
        speaker_embed: torch.Tensor,  # [B, T, F]
        text: torch.Tensor,  # [B, N]
        time: torch.Tensor,  # [B] or scalar
        drop_audio_cond: bool,
        drop_text: bool,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:  # [B, T, F]
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # Compute adaptive time conditioning and positional text embedding.
        t = self.time_embed(time)  # [B, dim]
        text_embed = self.text_embed(
            text, seq_len, drop_text=drop_text
        )  # [B, T, text_dim]

        # Project all conditioning signals into the model dimension.
        x = self.input_embed(
            x, cond, speaker_embed, text_embed, drop_audio_cond=drop_audio_cond
        )  # [B, T, dim]

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                x = cast(
                    torch.Tensor,
                    torch.utils.checkpoint.checkpoint(
                        self.ckpt_wrapper(block), x, t, mask, rope
                    ),
                )
            else:
                x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        return self.proj_out(x)  # [B, T, F]


# ─────────────────────────────────────────────────────────────────────────────
# CFM
# ─────────────────────────────────────────────────────────────────────────────


class CFM(nn.Module):
    """
    Conditional Flow Matching wrapper around DiT.

    Extends the original FlowSE CFM by requiring a speaker_embed tensor that
    is passed through to every DiT forward call.  All other behaviour
    (classifier-free guidance, ODE sampling, mel-spec conversion) is unchanged.

    Args:
        transformer:      DiT instance that accepts speaker_embed.
        sigma:            Noise level added to the optimal-transport path.
        odeint_kwargs:    kwargs forwarded to torchdiffeq.odeint during sampling.
        audio_drop_prob:  Probability of zeroing cond during CFG training.
        cond_drop_prob:   Probability of dropping both cond and text.
        num_channels:     Mel frequency bins F (inferred from mel_spec if None).
        mel_spec_module:  Custom MelSpec module (uses MelSpec(**mel_spec_kwargs) if None).
        mel_spec_kwargs:  kwargs for the default MelSpec constructor.
        vocab_char_map:   Optional character-to-index mapping for text tokenisation.

    forward input:
        inp:           [B, T, F] or [B, n_samples]  — noisy mel or waveform
        clean:         [B, T, F] or [B, n_samples]  — clean mel or waveform
        speaker_embed: [B, T, F]                    — speaker embedding
        text:          [B, N] or list[str]

    forward output:
        loss:  scalar MSE between predicted and ground-truth flow
        cond:  [B, T, F]  — conditioning mel (for logging)
        pred:  [B, T, F]  — predicted flow vector (for logging)
    """

    def __init__(
        self,
        *,
        transformer: DiT,
        sigma: float = 0.0,
        odeint_kwargs: dict[str, Any],
        audio_drop_prob: float = 0.0,
        cond_drop_prob: float = 0.0,
        num_channels: int | None = None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict[str, Any] | None = None,
        vocab_char_map: dict[str, int] | None = None,
    ) -> None:
        super().__init__()

        mel_spec_module = cast(nn.Module, mel_spec_module)
        num_channels = cast(int, num_channels)
        self.mel_spec = default(mel_spec_module, MelSpec(**(mel_spec_kwargs or {})))
        self.num_channels = default(num_channels, self.mel_spec.n_mel_channels)

        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob
        self.transformer = transformer
        self.dim = transformer.dim
        self.sigma = sigma
        self.odeint_kwargs = odeint_kwargs
        self.vocab_char_map = vocab_char_map

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _prepare_text(
        self,
        text: torch.Tensor | list[str],
        batch: int,
    ) -> torch.Tensor:
        """Tokenise a list of strings or pass through a pre-tokenised tensor."""
        if isinstance(text, list):
            if self.vocab_char_map is not None:
                text = list_str_to_idx(text, self.vocab_char_map).to(self.device)
            else:
                text = list_str_to_tensor(text).to(self.device)
            assert text.shape[0] == batch
        return text  # type: ignore[return-value]

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,  # [B, T, F] or [B, n_samples]
        speaker_embed: torch.Tensor,  # [B, T, F]
        text: torch.Tensor | list[str],
        *,
        steps: int = 32,
        cfg_strength: float = 1.0,
        vocoder: nn.Module | None = None,
        no_ref_audio: bool = False,
        drop_text: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Draw a sample via ODE integration from Gaussian noise to clean mel.

        Args:
            cond:          Noisy conditioning mel or raw waveform  [B, T, F] | [B, n_samples].
            speaker_embed: Pre-computed speaker embedding          [B, T, F].
            text:          Transcription tokens or raw strings.
            steps:         Number of Euler integration steps.
            cfg_strength:  Classifier-free guidance scale
                           (1.0 = no guidance, >1.0 = stronger guidance).
            vocoder:       Optional vocoder module converting mel -> waveform.
            no_ref_audio:  If True, zero out the conditioning signal entirely.
            drop_text:     If True, drop text conditioning (fully unconditional).

        Returns:
            out:        Sampled mel [B, T, F], or waveform [B, n_samples] if vocoder given.
            trajectory: Full ODE trajectory stacked along dim 0  [(steps+1), B, T, F].
        """
        self.eval()

        if cond.ndim == 2:
            cond = einops.rearrange(self.mel_spec(cond), "b f t -> b t f")
            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)
        batch = cond.shape[0]
        text = self._prepare_text(text, batch)

        if no_ref_audio:
            cond = torch.zeros_like(cond)

        mask = None

        def fn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            # Conditional prediction.
            pred = self.transformer(
                x=x,
                cond=cond,
                speaker_embed=speaker_embed,
                text=text,
                time=t,
                mask=mask,
                drop_audio_cond=False,
                drop_text=drop_text,
            )
            if cfg_strength < 1e-5:
                return pred
            # Unconditional prediction for CFG.
            null_pred = self.transformer(
                x=x,
                cond=cond,
                speaker_embed=speaker_embed,
                text=text,
                time=t,
                mask=mask,
                drop_audio_cond=True,
                drop_text=True,
            )
            return pred + (pred - null_pred) * cfg_strength

        y0 = torch.randn_like(cond)
        t = torch.linspace(0, 1, steps + 1, device=self.device, dtype=cond.dtype)
        trajectory = cast(torch.Tensor, odeint(fn, y0, t, **self.odeint_kwargs))
        out = trajectory[-1]  # [B, T, F]

        if vocoder is not None:
            out = vocoder(einops.rearrange(out, "b t f -> b f t"))

        return out, trajectory

    def forward(
        self,
        inp: torch.Tensor,  # [B, T, F] or [B, n_samples]
        clean: torch.Tensor,  # [B, T, F] or [B, n_samples]
        speaker_embed: torch.Tensor,  # [B, T, F]
        text: torch.Tensor | list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Convert raw waveforms to mel spectrograms if necessary.
        if inp.ndim == 2:
            inp = einops.rearrange(self.mel_spec(inp), "b f t -> b t f")
            clean = einops.rearrange(self.mel_spec(clean), "b f t -> b t f")
            assert inp.shape[-1] == self.num_channels

        batch, _, dtype = *inp.shape[:2], inp.dtype
        text = self._prepare_text(text, batch)

        # Build the optimal-transport interpolant φ_t = (1-t)*x0 + t*x1.
        x1 = clean
        x0 = torch.randn_like(x1)
        time = torch.rand((batch,), dtype=dtype, device=self.device)

        # Unsqueeze time for broadcasting against [B, T, F].
        t = einops.rearrange(time, "b -> b 1 1")
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0  # ground-truth flow vector

        cond = inp

        # Classifier-free guidance dropout.
        drop_audio_cond = random() < self.audio_drop_prob
        if random() < self.cond_drop_prob:
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        pred = self.transformer(
            x=φ,
            cond=cond,
            speaker_embed=speaker_embed,
            text=text,
            time=time,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
        )  # [B, T, F]

        loss = F.mse_loss(pred, flow, reduction="none")
        return loss.mean(), cond, pred


# ─────────────────────────────────────────────────────────────────────────────
# ModelConfig
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class ModelConfig:
    # STFT / frequency
    n_freqs: int  # n_fft // 2 + 1; frequency bins in the STFT domain
    n_fft: int
    emb_dim: int  # internal channel dimension of the mixture encoder
    eps: float = 1e-5
    n_channels: int = 1

    # Enrollment encoder (TFGridNet_encoder)
    enc_stride: int
    enc_n_blocks: int

    # Enrollment head (GridNetBlockAttnHead — pos/neg fusion)
    head_layer_num: int
    head_refine_layer_num: int
    head_fusion_shortcut: list[bool]
    head_cut_pos: bool = True
    head_return_clean_dvec: bool = False

    # Cross-attention (TFGridNetBlockAttn)
    attn_n_head: int = 4
    attn_approx_qk_dim: int = 512

    # DiT
    dit_dim: int  # transformer model dimension
    dit_depth: int = 8
    dit_heads: int = 8
    dit_dim_head: int = 64
    dit_dropout: float = 0.1
    dit_ff_mult: int = 4
    dit_mel_dim: int = 100  # F; must equal n_freqs
    dit_text_num_embeds: int = 256
    dit_text_dim: int | None = None
    dit_conv_layers: int = 0
    dit_long_skip_connection: bool = False
    dit_checkpoint_activations: bool = False

    # CFM
    cfm_sigma: float = 0.0
    cfm_audio_drop_prob: float = 0.0
    cfm_cond_drop_prob: float = 0.0
    cfm_odeint_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"method": "euler"}
    )
    cfm_mel_spec_kwargs: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# PNAttentionFlow
# ─────────────────────────────────────────────────────────────────────────────


class PNAttentionFlow(nn.Module):
    """
    Full target-speaker enhancement pipeline combining:

      1. Enrollment branch  — encodes positive/negative reference waveforms into
                              a contrastive speaker embedding via TFGridNet_encoder
                              and GridNetBlockAttnHead.
      2. Mixture encoder    — projects the STFT of the noisy mixture into the
                              same embedding space via Conv2d.
      3. Cross-attention    — conditions the mixture embedding on the enrollment
                              embedding (TFGridNetBlockAttn), yielding a
                              speaker-aware representation.
      4. CFM + DiT          — conditional flow matching model whose DiT backbone
                              receives the speaker embedding through InputEmbedding,
                              performing the actual speech enhancement.

    The speaker embedding produced in step 3 is reduced over the channel axis
    (mean) to obtain a [B, T, F] tensor that is directly compatible with
    InputEmbedding without any additional projection, because F == dit_mel_dim
    is guaranteed by construction.

    Args:
        conf:           ModelConfig dataclass holding all hyperparameters.
        vocab_char_map: Optional character-to-index map for text tokenisation.
    """

    def __init__(
        self,
        conf: ModelConfig,
        vocab_char_map: dict[str, int] | None = None,
    ) -> None:
        super().__init__()

        # ── Enrollment branch ────────────────────────────────────────────────
        # Encodes both positive (target) and negative (distractor) reference
        # waveforms into frame-level embeddings, then fuses them.
        self.enrollment_encoder = TFGridNet_encoder(
            num_ch=conf.n_channels,
            n_fft=conf.n_fft,
            stride=conf.enc_stride,
            num_blocks=conf.enc_n_blocks,
            binaural=(conf.n_channels % 2 == 0),
        )
        self.enrollment_head = GridNetBlockAttnHead(
            layer_num=conf.head_layer_num,
            pooling_size=1,
            stride=1,
            return_clean_dvec=conf.head_return_clean_dvec,
            out_dim=0,
            refine_layer_num=conf.head_refine_layer_num,
            fusion_shortcut=conf.head_fusion_shortcut,
            cut_pos=conf.head_cut_pos,
        )

        # ── Mixture encoder ──────────────────────────────────────────────────
        # Normalises the mixture by its per-sample std, computes the STFT, and
        # projects (real, imag) channels into emb_dim via a Conv2d.
        self.stft = STFT(
            n_fft=conf.n_fft,
            hop_length=conf.enc_stride,
            win_length=conf.n_fft,
        )
        self.mixture_proj = nn.Sequential(
            nn.Conv2d(
                in_channels=conf.n_channels * 2,  # real + imag per input channel
                out_channels=conf.emb_dim,
                kernel_size=(3, 3),
                padding=(1, 1),
            ),
            nn.GroupNorm(1, conf.emb_dim, eps=conf.eps),
        )

        # ── Cross-attention ──────────────────────────────────────────────────
        # Conditions the mixture embedding on the enrollment embedding so that
        # the resulting representation is specific to the target speaker.
        self.cross_attention = TFGridNetBlockAttn(
            emb_dim=conf.emb_dim,
            n_freqs=conf.n_freqs,
            n_head=conf.attn_n_head,
            approx_qk_dim=conf.attn_approx_qk_dim,
            eps=conf.eps,
        )

        # ── CFM (contains DiT) ───────────────────────────────────────────────
        # The DiT backbone receives speaker_embed through InputEmbedding.
        # dit_mel_dim must equal n_freqs so that the speaker embedding
        # [B, T, F] is directly compatible with (x, cond) without re-projection.
        assert conf.dit_mel_dim == conf.n_freqs, (
            f"dit_mel_dim ({conf.dit_mel_dim}) must equal n_freqs ({conf.n_freqs}) "
            "so that the speaker embedding F dimension is consistent."
        )
        transformer = DiT(
            dim=conf.dit_dim,
            depth=conf.dit_depth,
            heads=conf.dit_heads,
            dim_head=conf.dit_dim_head,
            dropout=conf.dit_dropout,
            ff_mult=conf.dit_ff_mult,
            mel_dim=conf.dit_mel_dim,
            text_num_embeds=conf.dit_text_num_embeds,
            text_dim=conf.dit_text_dim,
            conv_layers=conf.dit_conv_layers,
            long_skip_connection=conf.dit_long_skip_connection,
            checkpoint_activations=conf.dit_checkpoint_activations,
        )
        self.cfm = CFM(
            transformer=transformer,
            sigma=conf.cfm_sigma,
            odeint_kwargs=conf.cfm_odeint_kwargs,
            audio_drop_prob=conf.cfm_audio_drop_prob,
            cond_drop_prob=conf.cfm_cond_drop_prob,
            mel_spec_kwargs=conf.cfm_mel_spec_kwargs,
            vocab_char_map=vocab_char_map,
        )

    def encode_enrollment(
        self,
        positive: torch.Tensor,  # [B, n_channels, n_samples]
        negative: torch.Tensor,  # [B, n_channels, n_samples]
    ) -> torch.Tensor:  # [B, C, T_pos, F]
        """
        Encode positive/negative reference waveforms into a fused speaker embedding.

        The enrollment encoder maps each reference to frame-level features, and
        the enrollment head fuses the positive and negative embeddings into a
        single contrastive representation.  Only the T_pos frames are retained
        because the downstream cross-attention expects the conditioning length to
        match the positive reference length.

        Args:
            positive: Target-speaker reference waveform  [B, n_channels, n_samples].
            negative: Distractor-speaker reference       [B, n_channels, n_samples].

        Returns:
            Fused enrollment embedding  [B, C, T_pos, F].
        """
        # TFGridNet_encoder expects sequence-first layout: [B, n_samples, n_channels].
        pos_emb = self.enrollment_encoder(
            einops.rearrange(positive, "b c n -> b n c")
        )  # [B, C, T_pos, F]
        neg_emb = self.enrollment_encoder(
            einops.rearrange(negative, "b c n -> b n c")
        )  # [B, C, T_neg, F]

        # enrollment_head concatenates pos and neg along the time axis internally;
        # only the first T_pos frames (corresponding to the positive reference) are kept.
        fused = self.enrollment_head(pos_emb, neg_emb)  # [B, C, T_pos+T_neg, F]
        fused = fused[:, :, : pos_emb.shape[2], :]  # [B, C, T_pos, F]
        return fused

    def encode_mixture(
        self,
        mixture: torch.Tensor,  # [B, n_channels, n_samples]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Normalise, STFT, and project the noisy mixture waveform.

        Per-sample std normalisation is applied before the STFT so the model
        operates on a unit-scale signal.  The std is returned so the decoder
        can restore the original amplitude.

        Args:
            mixture: Noisy mixture waveform  [B, n_channels, n_samples].

        Returns:
            mix_emb: Projected mixture embedding  [B, emb_dim, T, F].
            std:     Per-sample std               [B, 1, 1].
        """
        std = mixture.std(dim=(1, 2), keepdim=True).clamp(min=1e-8)  # [B, 1, 1]
        _, _, real, imag, _ = self.stft(mixture / std)  # each [B, n_channels, F, T]

        # Concatenate real and imag along the channel axis, then swap F and T
        # so that Conv2d sees spatial layout [B, 2C, T, F].
        spec = einops.rearrange(
            torch.cat([real, imag], dim=1),
            "b c f t -> b c t f",
        )  # [B, 2*n_channels, T, F]
        mix_emb = self.mixture_proj(spec)  # [B, emb_dim, T, F]
        return mix_emb, std

    def extract_speaker_embed(
        self,
        mixture: torch.Tensor,  # [B, n_channels, n_samples]
        positive: torch.Tensor,  # [B, n_channels, n_samples]
        negative: torch.Tensor,  # [B, n_channels, n_samples]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Produce the speaker embedding fed into InputEmbedding.

        Runs the full encoding pipeline:
          mixture encoder -> cross-attention conditioned on enrollment -> channel mean.

        The mean over emb_dim collapses [B, emb_dim, T, F] to [B, T, F], which is
        directly compatible with the (x, cond) tensors inside DiT because
        F == dit_mel_dim is guaranteed by ModelConfig.

        Args:
            mixture:  Noisy mixture waveform         [B, n_channels, n_samples].
            positive: Target-speaker reference       [B, n_channels, n_samples].
            negative: Distractor-speaker reference   [B, n_channels, n_samples].

        Returns:
            speaker_embed: Speaker-conditioned embedding  [B, T, F].
            std:           Per-sample std of the mixture  [B, 1, 1].
        """
        mix_emb, std = self.encode_mixture(mixture)  # [B, emb_dim, T, F]
        enrollment_emb = self.encode_enrollment(positive, negative)  # [B, C, T_pos, F]

        # Cross-attention conditions the mixture embedding on the enrollment embedding,
        # yielding a speaker-aware mixture representation.
        attended = self.cross_attention(mix_emb, enrollment_emb)  # [B, emb_dim, T, F]

        # Mean over the channel axis gives [B, T, F], directly compatible with
        # InputEmbedding without any additional learned projection.
        speaker_embed = einops.reduce(attended, "b c t f -> b t f", "mean")
        return speaker_embed, std

    def forward(
        self,
        mixture: torch.Tensor,  # [B, n_channels, n_samples]
        positive: torch.Tensor,  # [B, n_channels, n_samples]
        negative: torch.Tensor,  # [B, n_channels, n_samples]
        clean: torch.Tensor,  # [B, T, F] or [B, n_samples]
        text: torch.Tensor | list[str],  # [B, N] or list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full training forward pass.

        Extracts the speaker embedding from the enrollment branch and mixture
        encoder, then delegates the flow matching loss computation to CFM.

        The mixture waveform serves a dual role: it is the source for speaker
        embedding extraction (via STFT + cross-attention) and the noisy
        conditioning input to CFM.  For mono inputs (n_channels == 1) the
        channel dim is squeezed before being passed to CFM, which expects
        either [B, n_samples] or [B, T, F].

        Args:
            mixture:  Noisy mixture waveform          [B, n_channels, n_samples].
            positive: Target-speaker reference        [B, n_channels, n_samples].
            negative: Distractor-speaker reference    [B, n_channels, n_samples].
            clean:    Ground-truth clean mel or wave  [B, T, F] | [B, n_samples].
            text:     Transcription tokens or strings [B, N]    | list[str].

        Returns:
            loss: Scalar MSE between predicted and ground-truth flow.
            cond: Conditioning mel used inside CFM  [B, T, F]  (for logging).
            pred: Predicted flow vector             [B, T, F]  (for logging).
        """
        speaker_embed, _ = self.extract_speaker_embed(mixture, positive, negative)

        # CFM expects [B, n_samples] for raw waveform input; squeeze the mono
        # channel dim so the shape contract is satisfied.
        inp = einops.rearrange(mixture, "b 1 n -> b n")
        return self.cfm(inp=inp, clean=clean, speaker_embed=speaker_embed, text=text)

    @torch.no_grad()
    def sample(
        self,
        mixture: torch.Tensor,  # [B, n_channels, n_samples]
        positive: torch.Tensor,  # [B, n_channels, n_samples]
        negative: torch.Tensor,  # [B, n_channels, n_samples]
        text: torch.Tensor | list[str],  # [B, N] or list[str]
        *,
        steps: int = 32,
        cfg_strength: float = 1.0,
        vocoder: nn.Module | None = None,
        no_ref_audio: bool = False,
        drop_text: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full inference pass.

        Extracts the speaker embedding, then runs ODE-based sampling inside CFM
        to produce the enhanced clean mel (or waveform if a vocoder is provided).

        Args:
            mixture:      Noisy mixture waveform           [B, n_channels, n_samples].
            positive:     Target-speaker reference         [B, n_channels, n_samples].
            negative:     Distractor-speaker reference     [B, n_channels, n_samples].
            text:         Transcription tokens or strings  [B, N] | list[str].
            steps:        Number of Euler ODE steps.
            cfg_strength: Classifier-free guidance scale
                            (1.0 = no guidance, >1.0 = stronger guidance).
            vocoder:      Optional vocoder converting mel -> waveform.
            no_ref_audio: If True, zero out the noisy conditioning signal.
            drop_text:    If True, run fully unconditional (no text guidance).

        Returns:
            out:        Enhanced mel [B, T, F], or waveform [B, n_samples] if vocoder given.
            trajectory: Full ODE trajectory  [(steps+1), B, T, F].
        """
        speaker_embed, _ = self.extract_speaker_embed(mixture, positive, negative)

        # Squeeze mono channel dim before passing to CFM.sample, which expects
        # either [B, n_samples] (raw waveform) or [B, T, F] (mel).
        cond = einops.rearrange(mixture, "b 1 n -> b n")
        return self.cfm.sample(
            cond=cond,
            speaker_embed=speaker_embed,
            text=text,
            steps=steps,
            cfg_strength=cfg_strength,
            vocoder=vocoder,
            no_ref_audio=no_ref_audio,
            drop_text=drop_text,
        )

    @classmethod
    def load_enrollment_branch(
        cls,
        model: PNAttentionFlow,
        checkpoint_path: str,
        *,
        freeze: bool = True,
        strict: bool = False,
        device: torch.device | str = "cpu",
    ) -> tuple[list[str], list[str]]:
        """
        Load enrollment branch weights from a Tar_Model checkpoint into a
        PNAttentionFlow instance, with optional parameter freezing.

        Tar_Model saves the enrollment branch under two key prefixes:
          - 'siamese.*'       -> PNAttentionFlow.enrollment_encoder.*
          - 'encoder_head.*'  -> PNAttentionFlow.enrollment_head.*

        All other keys (conv, attention_block, dual_mdl, deconv, etc.) belong
        to the extraction branch and are intentionally ignored, since the
        decoder architecture differs.

        Args:
            model:           PNAttentionFlow instance to load weights into.
            checkpoint_path: Path to the Tar_Model checkpoint file.
                             Accepts both raw state dicts and wrapped checkpoints
                             of the form {'state_dict': ...} or {'siamese': ...,
                             'encoder_head': ...} (encoder-only saves).
            freeze:          If True, requires_grad is set to False for all
                             enrollment branch parameters after loading, so they
                             are excluded from optimiser updates.  Set to False
                             to fine-tune the enrollment branch jointly.
            strict:          Passed to load_state_dict for each sub-module.
                             False allows partial loads when architecture details
                             differ slightly between checkpoint and current model.
            device:          Device to map checkpoint tensors onto before loading.

        Returns:
            missing_keys:    Keys present in the remapped state dict but absent
                             in the model (from load_state_dict).
            unexpected_keys: Keys present in the model but absent in the remapped
                             state dict (from load_state_dict).
        """
        raw = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Unwrap common checkpoint wrapper formats.
        if "state_dict" in raw:
            # Full model checkpoint saved as {'state_dict': model.state_dict()}.
            state_dict: dict[str, torch.Tensor] = raw["state_dict"]
        elif "siamese" in raw and "encoder_head" in raw:
            # Encoder-only checkpoint saved via Tar_Model.encoder_state_dict().
            # Flatten into a single dict with the original key prefixes restored.
            state_dict = {
                **{f"siamese.{k}": v for k, v in raw["siamese"].items()},
                **{f"encoder_head.{k}": v for k, v in raw["encoder_head"].items()},
            }
        else:
            state_dict = raw

        # Remap Tar_Model key prefixes to PNAttentionFlow key prefixes.
        PREFIX_MAP = {
            "siamese.": "enrollment_encoder.",
            "encoder_head.": "enrollment_head.",
        }

        remapped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            for src_prefix, dst_prefix in PREFIX_MAP.items():
                if key.startswith(src_prefix):
                    remapped[dst_prefix + key[len(src_prefix) :]] = value
                    break
            # Keys that don't match any prefix (extraction branch) are dropped.

        # Load into each sub-module independently so strict can be applied
        # per module without being tripped up by the other module's keys.
        enrollment_encoder_sd = {
            k[len("enrollment_encoder.") :]: v
            for k, v in remapped.items()
            if k.startswith("enrollment_encoder.")
        }
        enrollment_head_sd = {
            k[len("enrollment_head.") :]: v
            for k, v in remapped.items()
            if k.startswith("enrollment_head.")
        }

        enc_result = model.enrollment_encoder.load_state_dict(
            enrollment_encoder_sd, strict=strict
        )
        head_result = model.enrollment_head.load_state_dict(
            enrollment_head_sd, strict=strict
        )

        # Freeze or unfreeze the enrollment branch parameters.
        # Freezing is the default because the encoder is loaded from a
        # pre-trained Tar_Model checkpoint and the decoder architecture has
        # changed; only the CFM/DiT stack needs to be trained from scratch.
        enrollment_modules: list[nn.Module] = [
            model.enrollment_encoder,
            model.enrollment_head,
        ]
        for module in enrollment_modules:
            for param in module.parameters():
                param.requires_grad = not freeze

        status = "frozen" if freeze else "unfrozen (trainable)"
        print(f"[load_enrollment_branch] enrollment branch {status}")

        missing_keys = enc_result.missing_keys + head_result.missing_keys
        unexpected_keys = enc_result.unexpected_keys + head_result.unexpected_keys

        if missing_keys:
            print(
                f"[load_enrollment_branch] missing keys ({len(missing_keys)}): {missing_keys}"
            )
        if unexpected_keys:
            print(
                f"[load_enrollment_branch] unexpected keys ({len(unexpected_keys)}): {unexpected_keys}"
            )

        return missing_keys, unexpected_keys
