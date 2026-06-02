import torch
import torch.nn as nn

from ..modules import Attention, FeedForward


class AdaLayerNormZero(nn.Module):
    """
    adaLN-Zero block for DiT Block

    Linears with SiLU activation function: Shift/Scale/Gate
    """

    def __init__(self, dim: int):
        super().__init__()

        self.activation = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 3)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """adaLN-Zero forward
        Args:
            x: [B, T, dim] Tensor
            cond: [B, T, dim] Tensor

        Returns:
            Normalized x: [B, T, dim] Tensor
            Gate: [B, T, dim] Tensor
        """
        cond = self.linear(cond)
        cond = self.activation(cond)
        shift, scale, gate = torch.chunk(cond, chunks=3, dim=1)

        x = self.norm(x) * (1 + scale) + shift
        return x, gate


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        dim_head: int,
        ff_mult: int,
        dropout: float,
    ) -> None:
        """
        DiT Block for Diffusion Transformer,
        which consisted with attention layer, and FFN.

        Attention uses adaLN-zero as norm,
        FFN uses LayerNorm and adaLN-zero as norm.

        x: [B, T, dim]
        a: [B, T, 6 × dim]

                 ┌────────────────┐            ┌─────────────────────────┐
         x --->  │ Self-Attention │ ---> a --> │ Feed-Foward (× ff_mult) │  --> out
            │    └────────────────┘            └─────────────────────────┘   │
            │                                                                │
            └─────────────────────(Residual Connection)──────────────────────┘
        """

        super().__init__()

        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(
            dim=dim,
            n_head=n_head,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.ff_layernorm = nn.LayerNorm(
            dim,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.ff_adanorm = AdaLayerNormZero(dim)
        self.ff = FeedForward(
            in_channel=dim,
            mult=ff_mult,
            dropout=dropout,
            approximate="tanh",
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        rope: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm, attn_gate = self.attn_norm(x, cond=t)

        attn_output = self.attn(x=norm, mask=mask, rope=rope)
        x = x + attn_gate * attn_output

        norm = self.ff_layernorm(x)
        norm, ff_gate = self.ff_adanorm(norm, cond=t)
        ff_output = self.ff(x)
        out = x + ff_gate * ff_output

        return out
