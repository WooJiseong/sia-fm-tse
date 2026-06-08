"""DiT backbone for Mask2Flow-style STFT refinement."""

import torch
import torch.nn as nn
from x_transformers.x_transformers import RotaryEmbedding

from .modules import (
    ConditionEmbedding,
    FeedForward,
    InputEmbedding,
    MHAttention,
    MHCrossAttention,
    TimestepEmbedding,
)
from .transformer import AdaLN, AdaLNZero


class Mask2FlowDiTBlock(nn.Module):
    """Self-attention -> PN cross-attention -> FFN."""

    def __init__(
        self,
        dim: int,
        n_head: int,
        dim_head: int,
        ff_mult: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.attention = AdaLNZero(
            dim,
            layer=MHAttention(
                dim=dim,
                n_head=n_head,
                dim_head=dim_head,
                dropout=dropout,
            ),
        )
        self.cross_attention = AdaLNZero(
            dim,
            layer=MHCrossAttention(
                dim=dim,
                n_head=n_head,
                dim_head=dim_head,
                dropout=dropout,
            ),
        )
        self.ffn = AdaLNZero(
            dim,
            layer=FeedForward(
                dim=dim,
                mult=ff_mult,
                dropout=dropout,
                approximate="tanh",
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        rope: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.attention(x, c=t, mask=mask, rope=rope)
        x = self.cross_attention(x, c=t, context=c, rope=rope)
        x = self.ffn(x, c=t)
        return x


class Mask2FlowDiT(nn.Module):
    """DiT refiner that sees both the flow state and the original mixture.

    The flow starts from a coarse target estimate S0 and ends at the clean target
    source S. The mixture STFT Y is supplied as side information so the refiner
    can recover target components that the coarse estimate may have suppressed.
    """

    def __init__(
        self,
        *,
        dim: int,
        depth: int = 8,
        n_head: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1,
        ff_mult: int = 4,
        stft_dim: int,
        cond_in_ch: int = 64,
        cond_in_freq: int = 65,
        long_skip_connection: bool = False,
    ):
        super().__init__()
        self.time_emb = TimestepEmbedding(dim)
        self.input_emb = InputEmbedding(stft_dim * 2, out_dim=dim)
        self.cond_emb = ConditionEmbedding(
            in_ch=cond_in_ch,
            in_freq=cond_in_freq,
            out_dim=dim,
        )
        self.rotary_emb = RotaryEmbedding(dim_head)
        self.dim = dim
        self.stft_dim = stft_dim
        self.depth = depth
        self.transformer_blocks = nn.ModuleList(
            Mask2FlowDiTBlock(
                dim=dim,
                n_head=n_head,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(depth)
        )
        self.norm = AdaLN(dim)
        self.projection = nn.Linear(dim, stft_dim)
        self.long_skip_connection = (
            nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        )

    def forward(
        self,
        x: torch.Tensor,
        mixture: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = x.shape[1]
        t = self.time_emb(t)
        x = self.input_emb(x, mixture)
        c = self.cond_emb(c)
        rope = self.rotary_emb.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            x = block(x, t, c, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm(x, t)
        return self.projection(x)
