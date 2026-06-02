import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from x_transformers.x_transformers import apply_rotary_pos_emb


class FeedForward(nn.Module):
    """
    Simple feed-forward network with GELU activation
    """

    def __init__(
        self,
        in_channel: int,
        *,
        out_channel: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        approximate: str = "none",
    ):
        super().__init__()
        out_channel = out_channel if out_channel is not None else in_channel

        self.ff = nn.Sequential(
            nn.Linear(in_channel, in_channel * mult),
            nn.GELU(approximate=approximate),
            nn.Dropout(dropout),
            nn.Linear(in_channel * mult, out_channel),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class Attention(nn.Module):
    """Attention with RoPE"""

    def __init__(
        self,
        *,
        dim: int,
        n_head: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
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
        mask: torch.Tensor,
        rope: tuple[torch.Tensor, float] | None = None,
    ) -> torch.Tensor:
        query = self.linear_q(x)
        key = self.linear_k(x)
        value = self.linear_v(x)

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
        query = rearrange(query, "b n (h d) -> b h n d", h=self.n_head)
        key = rearrange(key, "b n (h d) -> b h n d", h=self.n_head)
        value = rearrange(value, "b n (h d) -> b h n d", h=self.n_head)

        attn_mask = (
            None
            if mask is None
            else rearrange(mask, "b n -> b h n d", h=self.n_head, d=self.dim)
        )

        x = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0,
            is_causal=False,
        )
        x = rearrange(x, "b h n d -> b n (h d)").to(query.dtype)
        out = self.out_proj(x)

        if mask is not None:
            mask = rearrange(mask, "b n -> b n hd", hd=self.n_head * self.dim)
            out = out.masked_fill(~mask, 0.0)

        return out
