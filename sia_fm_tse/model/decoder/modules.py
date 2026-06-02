"""universal modules referred to other modules"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from x_transformers.x_transformers import apply_rotary_pos_emb


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        out_dim: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        approximate: str = "none",
    ):
        """
        Position-wise feed-forward network with GELU activation.

        Args:
            dim:         input dimension
            out_dim:     output dimension (default: same as dim)
            mult:        hidden dim multiplier (hidden = dim × mult)
            dropout:     dropout rate
            approximate: GELU approximation method
        """
        super().__init__()
        out_dim = out_dim if out_dim is not None else dim

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(approximate=approximate),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] Tensor

        Returns:
            x: [B, N, out_dim] Tensor
        """
        return self.ff(x)


class MHAttention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        n_head: int = 8,
        dim_head: int = 64,
        dropout: float = 0,
    ):
        """
        Multi-head self-attention with RoPE support.

        Args:
            dim:      input/output dimension
            n_head:   number of attention heads
            dim_head: dimension per head (inner_dim = n_head × dim_head)
            dropout:  attention dropout rate
        """
        super().__init__()
        self.dim = dim
        self.n_head = n_head
        self.dropout = dropout

        self.inner_dim = dim_head * n_head
        self.linear_q = nn.Linear(dim, self.inner_dim)
        self.linear_k = nn.Linear(dim, self.inner_dim)
        self.linear_v = nn.Linear(dim, self.inner_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope: tuple[torch.Tensor, float] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    [B, N, D] Tensor
            mask: Optional [B, N] Tensor
            rope: Optional (freqs, xpos_scale) tuple from RotaryEmbedding

        Returns:
            out: [B, N, D] Tensor
        """
        query = self.linear_q(x)
        key = self.linear_k(x)
        value = self.linear_v(x)

        query = rearrange(query, "b n (h d) -> b h n d", h=self.n_head)
        key = rearrange(key, "b n (h d) -> b h n d", h=self.n_head)
        value = rearrange(value, "b n (h d) -> b h n d", h=self.n_head)

        # Apply rotary position embedding
        if rope is not None:
            freqs, xpos_scale = rope
            xpos_scale = xpos_scale if xpos_scale is not None else 1
            q_xpos_scale, k_xpos_scale = xpos_scale, 1 / xpos_scale

            # `apply_rotary_pos_emb` uses float scale factor;
            # but there is no type hint, so linter thinks it as type int.
            # => ignore.
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)  # type: ignore
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)  # type: ignore

        # Attention
        attn_mask = None if mask is None else rearrange(mask, "b n -> b 1 1 n")

        x = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            is_causal=False,
        )
        x = rearrange(x, "b h n d -> b n (h d)").to(value.dtype)
        out = self.out_proj(x)

        if mask is not None:
            mask = rearrange(mask, "b n -> b n 1")
            out = out.masked_fill(~mask, 0.0)

        return out


class SinusoidalPositionalEmbedding(torch.nn.Module):
    def __init__(self, dim: int):
        """
        Sinusoidal positional embedding.
        Encodes scalar values into a high-dimensional vector
        using sine and cosine functions at different frequencies.

        Args:
            dim: output embedding dimension (must be even)
        """
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, f"{self.__class__} requires dim to be even"

    def forward(self, x: torch.Tensor, scale: int = 1000):
        """
        Args:
            x: [B] Tensor
            scale: scaling factor applied before sin/cos (default: 1000)

        Returns:
            emb: [B, D] Tensor
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int = 256):
        """
        Timestep embedding for diffusion process.
        Encodes diffusion timestep t into a condition vector
        via sinusoidal encoding followed by a two-layer MLP.

        Args:
            dim: output embedding dimension
            freq_embed_dim: sinusoidal encoding dimension,
                            controls frequency resolution before MLP projection
        """
        super().__init__()
        self.sinusoidal = SinusoidalPositionalEmbedding(dim=freq_embed_dim)
        self.projection = nn.Sequential(
            nn.Linear(freq_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor):
        """
        Args:
            t: [B] Tensor

        Returns:
            t: [B, D] Tensor
        """
        h = self.sinusoidal(t).to(t.dtype)
        t = self.projection(h)
        return t


class ConvPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 31, groups: int = 16):
        """
        Convolutional positional embedding.
        Injects local positional information via depthwise-style Conv1d,
        instead of absolute sinusoidal or learned embeddings.

        Args:
            dim:         input/output channel dimension
            kernel_size: convolution kernel size (must be odd for symmetric padding)
            groups:      number of conv groups (depthwise-like when groups == dim)
        """
        super().__init__()
        assert kernel_size % 2 != 0, f"{self.__class__} requires kernel_size to be odd"
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] Tensor
            mask: Optional [B, N] Tensor (True = valid)

        Returns:
            out: [B, N, D] Tensor
        """
        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        x = rearrange(x, "b n d -> b d n")
        out = self.conv1d(x)
        out = rearrange(out, "b d n -> b n d")

        if mask is not None:
            out = out.masked_fill(~mask, 0.0)

        return out


class InputEmbedding(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        """
        Input embedding for multiple input tensors.
        Concatenates inputs along the feature dim, projects to out_dim,
        then adds convolutional positional embedding.

        Args:
            dim: sum of feature dimensions of each input tensor
            out_dim:     output embedding dimension
        """
        super().__init__()
        self.projection = nn.Linear(dim, out_dim)
        self.conv_pos_emb = ConvPositionalEmbedding(dim=out_dim)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            *inputs: [B, N, D_i] Tensors

        Returns:
            x: [B, N, out_dim] Tensor
        """
        x = torch.cat(inputs, dim=-1)
        x = self.projection(x)
        x = self.conv_pos_emb(x) + x
        return x
