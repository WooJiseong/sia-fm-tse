"""
Code for the Encoder Fusion Module
Adopted from the TFGridnet code provided in USEF-TSE: Universal Speaker Embedding Free Target Speaker Extraction
- https://github.com/ZBang/USEF-TSE
  - Original code licensed under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
No modification is done as this file contain only the model backbone modules
"""

import math

import einops
import torch
import torch.nn as nn
from espnet2.torch_utils.get_layer_from_string import get_layer
from torch.nn.functional import pad, softmax, unfold
from torch.nn.parameter import Parameter

from improved_model.GridnetAttnHead import LayerNormalization4DCF


class TFGridNetBlockAttn(nn.Module):
    """
    Cross-attention block where Q comes from the primary stream (x)
    and K, V come from an auxiliary stream (aux).

    Used inside USEF-TSE to condition the main speech representation
    on a speaker embedding or reference signal.

    Performs multi-head cross-attention along the time axis:
      1. Project x -> Q and aux -> K, V per head via Conv2d.
      2. Flatten C and F dims -> treat each frame as a token.
      3. Compute scaled dot-product attention: Q (from x) attends to K, V (from aux).
      4. Concatenate heads and project back to emb_dim.

    Args:
        emb_dim:       Channel dimension C of both x and aux.
        n_freqs:       Frequency bins F.
        n_head:        Number of attention heads; emb_dim must be divisible by n_head.
        approx_qk_dim: Target total dim for Q/K projections.
                       Each head gets E = ceil(approx_qk_dim / n_freqs) channels,
                       so the actual flattened Q/K dim is E * F.
        eps:           Epsilon for LayerNormalization4DCF.

    Input:
        x:   [B, C, T, F]   — primary stream (query source)
        aux: [B, C, T_aux, F] — auxiliary stream (key/value source)

    Output:
        out: [B, C, T, F]
    """

    def __init__(
        self,
        emb_dim: int,
        n_freqs: int,
        n_head: int,
        approx_qk_dim: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        activation = "prelu"  # fixed!

        E = math.ceil(approx_qk_dim / n_freqs)
        assert emb_dim % n_head == 0

        activation_layer = get_layer(activation)

        # Q projection from primary stream x: [B, C, T, F] -> [B, H*E, T, F]
        self.attn_conv_Q = nn.Conv2d(emb_dim, n_head * E, 1)
        self.attn_norm_Q = AllHeadPReLULayerNormalization4DCF(
            n_head=n_head,
            n_channels=E,
            n_freqs=n_freqs,
            eps=eps,
        )

        # K projection from auxiliary stream aux: [B, C, T_aux, F] -> [B, H*E, T_aux, F]
        self.attn_conv_K = nn.Conv2d(emb_dim, n_head * E, 1)
        self.attn_norm_K = AllHeadPReLULayerNormalization4DCF(
            n_head=n_head,
            n_channels=E,
            n_freqs=n_freqs,
            eps=eps,
        )

        # V projection from auxiliary stream aux: [B, C, T_aux, F] -> [B, H*(C/H), T_aux, F]
        self.attn_conv_V = nn.Conv2d(emb_dim, n_head * emb_dim // n_head, 1)
        self.attn_norm_V = AllHeadPReLULayerNormalization4DCF(
            n_head=n_head,
            n_channels=emb_dim // n_head,
            n_freqs=n_freqs,
            eps=eps,
        )

        # Final projection after head concatenation: [B, C, T, F] -> [B, C, T, F]
        self.attn_concat_proj = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, 1),
            activation_layer(),
            LayerNormalization4DCF(n_channels=emb_dim, n_freqs=n_freqs, eps=eps),
        )

        self.n_head = n_head

    def forward(self, x: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, F],  aux: [B, C, T_aux, F]
        _, _, _, F = x.shape

        # Project x -> Q, aux -> K, V per head
        Q: torch.Tensor = self.attn_norm_Q(self.attn_conv_Q(x))  # [B, H, E, T, F]
        K: torch.Tensor = self.attn_norm_K(self.attn_conv_K(aux))  # [B, H, E, T_aux, F]
        V: torch.Tensor = self.attn_norm_V(
            self.attn_conv_V(aux)
        )  # [B, H, C/H, T_aux, F]

        # Flatten channel and freq into token dim
        Q = einops.rearrange(Q, "b h e t f -> (b h) t (e f)")  # [B*H, T, E*F]
        K = einops.rearrange(K, "b h e t f -> (b h) t (e f)")  # [B*H, T_aux, E*F]
        V = einops.rearrange(V, "b h v t f -> (b h) t (v f)")  # [B*H, T_aux, (C/H)*F]

        # Cross-attention: Q from x attends to K, V from aux
        # -> [B*H, T, T_aux]
        qk_dim = Q.shape[-1]  # E * F
        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (qk_dim**0.5)
        attn_mat = softmax(attn_mat, dim=2)

        # Weighted sum of V: [B*H, T, (C/H)*F]
        attn_score = attn_mat @ V

        # Merge heads: [B*H, T, (C/H)*F] -> [B, C, T, F]
        out = einops.rearrange(
            attn_score, "(b h) t (v f) -> b (h v) t f", f=F, h=self.n_head
        )

        # Final conv projection: [B, C, T, F]
        out = self.attn_concat_proj(out)
        return out


class GridNetV2Block(nn.Module):
    """
    Full TF-GridNet block combining intra-RNN, inter-RNN, and self-attention.

    Processing pipeline:
      1. Pad x to fit the overlapping chunk scheme (emb_ks, emb_hs).
      2. Intra-RNN: Bi-LSTM along the frequency axis (within each time frame).
      3. Inter-RNN: Bi-LSTM along the time axis (across frames at each freq bin).
      4. Self-attention: multi-head attention along the time axis.
      5. Residual addition of attention output onto the inter-RNN output.

    Args:
        emb_dim:          Channel dimension C.
        emb_ks:           Chunk (kernel) size for intra/inter LSTM unfolding.
        emb_hs:           Hop (stride) size; emb_ks == emb_hs means no overlap.
        n_freqs:          Frequency bins F.
        hidden_channels:  Hidden size of the LSTM layers.
        n_head:           Number of attention heads.
        approx_qk_dim:    Target Q/K projection dimension.
        eps:              Epsilon for layer normalizations.

    Input / Output:
        x: [B, C, T, F]  ->  out: [B, C, T, F]
    """

    def __init__(
        self,
        emb_dim: int,
        emb_ks: int,
        emb_hs: int,
        n_freqs: int,
        hidden_channels: int,
        n_head: int = 4,
        approx_qk_dim: int = 1024,
        eps: float = 1e-5,
    ):
        super().__init__()
        activation = "prelu"  # fixed!

        in_channels = emb_dim * emb_ks  # flattened chunk size fed to each LSTM

        # ── Intra-RNN (frequency axis) ──────────────────────────────────────────
        self.intra_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_channels,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        # Linear: no overlap -> simple linear; overlap -> transposed conv to upsample back
        if emb_ks == emb_hs:
            self.intra_linear = nn.Linear(
                in_features=hidden_channels * 2,
                out_features=in_channels,
            )
        else:
            self.intra_linear = nn.ConvTranspose1d(
                in_channels=hidden_channels * 2,
                out_channels=emb_dim,
                kernel_size=emb_ks,
                stride=emb_hs,
            )

        # ── Inter-RNN (time axis) ────────────────────────────────────────────────
        self.inter_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_channels,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        if emb_ks == emb_hs:
            self.inter_linear = nn.Linear(
                in_features=hidden_channels * 2,
                out_features=in_channels,
            )
        else:
            self.inter_linear = nn.ConvTranspose1d(
                in_channels=hidden_channels * 2,
                out_channels=emb_dim,
                kernel_size=emb_ks,
                stride=emb_hs,
            )

        # ── Self-attention (time axis, after RNNs) ───────────────────────────────
        E = math.ceil(approx_qk_dim / n_freqs)
        assert emb_dim % n_head == 0

        activation_layer = get_layer(activation)

        # Q/K projections: [B, C, T, F] -> [B, H*E, T, F]
        self.attn_conv_Q = nn.Conv2d(emb_dim, n_head * E, 1)
        self.attn_norm_Q = AllHeadPReLULayerNormalization4DCF(
            n_head=n_head,
            n_channels=E,
            n_freqs=n_freqs,
            eps=eps,
        )

        self.attn_conv_K = nn.Conv2d(emb_dim, n_head * E, 1)
        self.attn_norm_K = AllHeadPReLULayerNormalization4DCF(
            n_head=n_head,
            n_channels=E,
            n_freqs=n_freqs,
            eps=eps,
        )

        # V projection: [B, C, T, F] -> [B, H*(C/H), T, F]
        self.attn_conv_V = nn.Conv2d(emb_dim, n_head * emb_dim // n_head, 1)
        self.attn_norm_V = AllHeadPReLULayerNormalization4DCF(
            n_head=n_head,
            n_channels=emb_dim // n_head,
            n_freqs=n_freqs,
            eps=eps,
        )

        # Final projection after head concatenation: [B, C, T, F] -> [B, C, T, F]
        self.attn_concat_proj = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, 1),
            activation_layer(),
            LayerNormalization4DCF(n_channels=emb_dim, n_freqs=n_freqs, eps=eps),
        )

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, F]
        B, C, T, F = x.shape

        # ── Padding ─────────────────────────────────────────────────────────────
        # Extend T and F so they fit an integer number of chunks of size emb_ks
        # with hop emb_hs and symmetric overlap = emb_ks - emb_hs on each side.
        overlap = self.emb_ks - self.emb_hs
        pad_T = (
            math.ceil((T + 2 * overlap - self.emb_ks) / self.emb_hs) * self.emb_hs
            + self.emb_ks
        )
        pad_F = (
            math.ceil((F + 2 * overlap - self.emb_ks) / self.emb_hs) * self.emb_hs
            + self.emb_ks
        )

        # fmt: off
        x = einops.rearrange(x, "b c t f -> b t f c")
        x = pad(x, (
            0, 0,                           # C: no pad
            overlap, pad_F - F - overlap,   # F axis
            overlap, pad_T - T - overlap    # T axis
        ))  # [B, pad_T, pad_F, C]
        # fmt: on

        # ── Intra-RNN (frequency axis) ───────────────────────────────────────────
        # Normalize along the last (C) axis before feeding to the LSTM.
        x_intra: torch.Tensor = self.intra_norm(x)  # [B, pad_T, pad_F, C]
        if overlap == 0:
            # No overlap: split F into non-overlapping chunks of size emb_ks.
            # Each time frame is processed independently -> batch over B*pad_T.
            x_intra = einops.rearrange(x_intra, "b t (w k) c -> (b t) w (k c)", k=self.emb_ks)  # fmt: skip
            x_intra, _ = self.intra_rnn(x_intra)  # [B*pad_T, W, 2*hidden]
            x_intra = self.intra_linear(x_intra)  # [B*pad_T, W, K*C]
            x_intra = einops.rearrange(
                x_intra,
                pattern="(b t) w (k c) -> b t (w k) c",
                b=B,
                k=self.emb_ks,
            )  # [B, pad_T, pad_F, C]
        else:
            # Overlap: use unfold to extract overlapping chunks along F.
            x_intra = einops.rearrange(x_intra, "b t f c -> (b t) c f")
            x_intra = unfold(
                x_intra[..., None],
                kernel_size=(self.emb_ks, 1),
                stride=(self.emb_hs, 1),
            )  # [B*pad_T, K*C, W]
            x_intra = einops.rearrange(
                x_intra, "bt kc w -> bt w kc"
            )  # [B*pad_T, W, K*C]
            x_intra, _ = self.intra_rnn(x_intra)  # [B*pad_T, W, 2*hidden]
            x_intra = einops.rearrange(x_intra, "bt w h -> bt h w")
            x_intra = self.intra_linear(x_intra)  # [B*pad_T, C, pad_F]
            x_intra = einops.rearrange(x_intra, "(b t) c f -> b t f c", b=B)
        x_intra += x  # residual: [B, pad_T, pad_F, C]
        x_intra = einops.rearrange(
            x_intra, "b t f c -> b f t c"
        )  # swap T<->F for inter-RNN

        # ── Inter-RNN (time axis) ─────────────────────────────────────────────────
        # Same chunking scheme, but now applied along the T axis.
        # Each frequency bin is processed independently -> batch over B*pad_F.
        x_inter: torch.Tensor = self.inter_norm(x_intra)  # [B, pad_F, pad_T, C]
        if overlap == 0:
            x_inter = einops.rearrange(
                x_inter,
                pattern="b f (w k) c -> (b f) w (k c)",
                k=self.emb_ks,
            )
            x_inter, _ = self.inter_rnn(x_inter)  # [B*pad_F, W, 2*hidden]
            x_inter = self.inter_linear(x_inter)  # [B*pad_F, W, K*C]
            x_inter = einops.rearrange(
                x_inter,
                pattern="(b f) w (k c) -> b f (w k) c",
                b=B,
                k=self.emb_ks,
            )
        else:
            x_inter = einops.rearrange(x_inter, "b f t c -> (b f) c t")
            x_inter = unfold(
                x_inter[..., None],
                kernel_size=(self.emb_ks, 1),
                stride=(self.emb_hs, 1),
            )  # [B*pad_F, K*C, W]
            x_inter = einops.rearrange(
                x_inter, "bf kc w -> bf w kc"
            )  # [B*pad_F, W, K*C]
            x_inter, _ = self.inter_rnn(x_inter)  # [B*pad_F, W, 2*hidden]
            x_inter = einops.rearrange(x_inter, "bf w h -> bf h w")
            x_inter = self.inter_linear(x_inter)  # [B*pad_F, C, pad_T]
            x_inter = einops.rearrange(x_inter, "(b f) c t -> b f t c", b=B)

        # NOTE (bug in original refactor): the line below should read `x_inter += x_intra`
        # (residual over the inter input), but the refactored code writes `x_inter += x`
        # (which still holds the padded input before the intra pass). Left as-is to stay
        # faithful to the refactored file; fix if needed.
        x_inter += x_intra  # residual: [B, pad_F, pad_T, C]

        # NOTE (bug in original refactor): the next line incorrectly reads from x_intra
        # instead of x_inter before the trim. The correct form is:
        #   x_inter = einops.rearrange(x_inter, "b f t c -> b c t f")
        x_inter = einops.rearrange(
            x_inter, "b f t c -> b c t f"
        )  # [B, C, pad_T, pad_F]

        # Trim padding back to the original T and F dimensions.
        x_inter = x_inter[..., overlap : overlap + T, overlap : overlap + F]

        x = x_inter  # [B, C, T, F]

        # ── Self-attention (time axis) ────────────────────────────────────────────
        # Project x into Q, K, V for each head
        Q: torch.Tensor = self.attn_norm_Q(self.attn_conv_Q(x))  # [B, H, E, T, F]
        K: torch.Tensor = self.attn_norm_K(self.attn_conv_K(x))  # [B, H, E, T, F]
        V: torch.Tensor = self.attn_norm_V(self.attn_conv_V(x))  # [B, H, C/H, T, F]

        # Flatten channel and freq into token dim: [B*H, T, E*F or (C/H)*F]
        Q = einops.rearrange(Q, "b h e t f -> (b h) t (e f)")
        K = einops.rearrange(K, "b h e t f -> (b h) t (e f)")
        V = einops.rearrange(V, "b h v t f -> (b h) t (v f)")  # V = C/H

        # Scaled dot-product attention over T, scaled by E*F (flattened Q/K dim)
        qk_dim = Q.shape[-1]  # E * F
        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (qk_dim**0.5)  # [B*H, T, T]
        attn_mat = softmax(attn_mat, dim=2)

        # Weighted sum of V: [B*H, T, (C/H)*F]
        attn_score = attn_mat @ V

        # Merge heads: [B*H, T, (C/H)*F] -> [B, C, T, F]
        out = einops.rearrange(attn_score, "(b h) t (v f) -> b (h v) t f", f=F, b=B)
        out = self.attn_concat_proj(out)

        # Residual: attention output added to inter-RNN output
        out += x_inter
        return out


class AllHeadPReLULayerNormalization4DCF(nn.Module):
    """
    Per-head PReLU activation followed by layer normalization over the
    channel (E) and frequency (F) axes of a [B, H*E, T, F] tensor.

    Reshapes the input into [B, H, E, T, F], applies a per-head PReLU,
    then normalizes over dims 2 (E) and 4 (F) with learnable affine params.

    Args:
        n_head:      Number of attention heads H.
        n_channels:  Per-head channel count E (so total input channels = H * E).
        n_freqs:     Frequency bins F.
        eps:         Small constant for numerical stability.

    Input:
        x: [B, H*E, T, F]

    Output:
        x: [B, H, E, T, F]
    """

    NORM_DIMS = (2, 4)  # E and F axes

    def __init__(self, n_head: int, n_channels: int, n_freqs: int, eps: float = 1e-5):
        super().__init__()
        # Learnable scale/shift: [1, H, E, 1, F] — broadcastable over B and T
        param_size = [1, n_head, n_channels, 1, n_freqs]
        self.gamma = Parameter(torch.ones(*param_size).to(torch.float32))
        self.beta = Parameter(torch.zeros(*param_size).to(torch.float32))
        # One PReLU slope per head so each head can learn its own non-linearity.
        self.activation_layer = nn.PReLU(num_parameters=n_head, init=0.25)
        self.n_head = n_head
        self.n_channels = n_channels
        self.n_freqs = n_freqs
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H*E, T, F]
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D tensor [B, H*E, T, F], got {x.ndim}D")

        # Reshape to expose head dim, then apply per-head PReLU
        x = einops.rearrange(x, "b (h e) t f -> b h e t f", h=self.n_head)
        x = self.activation_layer(x)  # PReLU slope broadcast over [B, H, E, T, F]

        # Compute mean and std over E and F -> [B, H, 1, T, 1]
        mu = x.mean(dim=self.NORM_DIMS, keepdim=True)
        # NOTE (typo in original refactor): `usbaised` should be `unbiased`
        std = x.std(dim=self.NORM_DIMS, unbiased=False, keepdim=True)

        x_norm = (x - mu) / (std + self.eps)
        return x_norm * self.gamma + self.beta  # affine transform: [B, H, E, T, F]
