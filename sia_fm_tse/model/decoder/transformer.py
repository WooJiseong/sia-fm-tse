"""Diffusion Transformer Implementation"""

# Refer to https://arxiv.org/pdf/2212.09748
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


class AdaLNZero(nn.Module):
    def __init__(self, dim: int, layer: nn.Module):
        """
        adaLN-Zero block

        Pre/Post-normalize layer input/output
        Projects condition Tensor c to factors: γ, β, α

                 ┌────────────────┐
         c --->  │ Linear w/ SiLU │ ---> γ, β, α
                 └────────────────┘

        with this factors,

                ┌─────────────────┐                      ┌───────────────┐
         x -->  │ LayerNorm(-> z) │ ---> (1+γ)z + β ---> │ `layer`(-> y) │ --> x + αy
            │   └─────────────────┘                      └───────────────┘  │
            │                                                               │
            └─────────────────────(Residual Connection)─────────────────────┘

        which would be, from now on, implmented as

               ┌──adaLN-zero──┐
         x --> │   `layer`    │ --> out
               └──────────────┘
        """
        super().__init__()

        # Condition projection layer
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 3)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        # Normalization
        self.layernorm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.layer = layer

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """adaLN-Zero forward
        Args:
            x: [B, N, D] Tensor
            c: [B, D] Tensor
            ...: etc layer arguments

        Returns:
            x: [B, N, D] Tensor
        """

        # Condition Projection
        # -> SiLU first is not error:
        #      https://github.com/facebookresearch/DiT/blob/main/models.py#L113
        c = self.linear(self.silu(c))
        γ, β, α = torch.chunk(c, chunks=3, dim=1)
        γ = γ.unsqueeze(1)
        β = β.unsqueeze(1)
        α = α.unsqueeze(1)

        # Pre-norm
        z = self.layernorm(x)
        z = (1 + γ) * z + β

        # Layer
        y = self.layer(z, **kwargs)

        # Post-norm
        out = x + α * y

        return out


class AdaLN(nn.Module):
    def __init__(self, dim: int):
        """
        Adaptive Layer Normalization (adaLN) without gating.
        Used as final modulation before output projection.

        Compared to AdaLNZero, this omits α (no residual gating)
        and returns the modulated tensor directly instead of x + αy.

        Args:
            dim: input/condition dimension
        """
        super().__init__()

        # Condition projection layer
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        # Normalization
        self.layernorm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] Tensor
            c: [B, D] Tensor

        Returns:
            z: [B, N, D] modulated Tensor
        """
        c = self.linear(self.silu(c))
        γ, β = torch.chunk(c, chunks=2, dim=1)
        γ = γ.unsqueeze(1)
        β = β.unsqueeze(1)

        z = self.layernorm(x)
        z = (1 + γ) * z + β
        return z


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
        DiT Block: Self-Attention → FFN → Cross-Attention, each wrapped in adaLN-Zero.

        x: [B, T, D]  main sequence (mel frames)
        c: [B, T_c, D]  encoder condition (may have different length T_c)
        t: [B, D]       timestep embedding

                 ┌───adaLN-zero───┐     ┌───────adaLN-zero─────────┐     ┌────────adaLN-zero─────────┐
         x --->  │ Self-Attention │ --> │ Feed-Forward (× ff_mult) │ --> │ Cross-Attention (c as KV) │ --> out
            │    └────────────────┘     └──────────────────────────┘     └───────────────────────────┘  │
            │                                                                                           │
            └──────────────────────────────────(Residual)───────────────────────────────────────────────┘
        """

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
        self.ffn = AdaLNZero(
            dim,
            layer=FeedForward(
                dim=dim,
                mult=ff_mult,
                dropout=dropout,
                approximate="tanh",
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

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        rope: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    [B, T, D] main sequence
            t:    [B, D] timestep embedding (adaLN condition)
            c:    [B, T_c, D] encoder condition — T_c may differ from T
            mask: Optional [B, T] boolean Tensor for self-attention (True = valid)
            rope: Optional (freqs, xpos_scale)

        Returns:
            x: [B, T, D]
        """
        x = self.attention(x, c=t, mask=mask, rope=rope)
        x = self.ffn(x, c=t)
        x = self.cross_attention(x, c=t, context=c, rope=rope)
        return x


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int = 8,
        n_head: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1,
        ff_mult: int = 4,
        mel_dim: int = 100,
        cond_in_ch: int = 64,
        cond_in_freq: int = 65,
        long_skip_connection: bool = False,
    ):
        """
        Diffusion Transformer (DiT) — https://arxiv.org/pdf/2212.09748

        x and m are jointly embedded via InputEmbedding (mel_dim*2 -> dim).
        c (raw encoder output) is independently embedded via ConditionEmbedding
        (cond_in_ch * cond_in_freq -> dim), which handles channel/freq flattening.
        c may have a different sequence length T_c; cross-attention handles this naturally.

        c: [B, C, T_c, F]  <- raw encoder output (T_c may differ from T)
        m: [B, T, mel_dim] <- mixture mel-spectrogram
        x: [B, T, mel_dim] <- noisy mel-spectrogram
        t: [B]

                   ╭──────────────────────────╮
         x, m -->  │ InputEmbedding           │ --> x_emb [B, T, D]
                   │ (mel_dim*2 -> dim)       │
                   ╰──────────────────────────╯

                 ╭──────────────────────────╮
         c  -->  │ ConditionEmbedding       │ --> c_emb [B, T_c, D]
                 │ (C*F -> dim, ConvPosEmb) │
                 ╰──────────────────────────╯

                 ┌───────────────────┐
         t --->  │ Timestep Encoding │
                 └───────────────────┘
                          ┌────┴────────────────────────┐
                  ┌───────────┐                     ┌───────┐     ┌────────┐
        x_emb --> │ DiT Block │ × depth ----------> │ adaLN │ --> │ Linear │ --> out
                  └───────────┘                 │   └───────┘     └────────┘
        c_emb ─────────┘ (cross-attn, T_c≠T OK) │
        x_emb ──────────────────(+Skip)─────────┘
        """
        super().__init__()
        self.time_emb = TimestepEmbedding(dim)
        self.input_emb = InputEmbedding(mel_dim * 2, out_dim=dim)
        self.cond_emb = ConditionEmbedding(
            in_ch=cond_in_ch, in_freq=cond_in_freq, out_dim=dim
        )
        self.rotary_emb = RotaryEmbedding(dim_head)
        self.dim = dim
        self.mel_dim = mel_dim
        self.depth = depth
        self.transformer_blocks = nn.ModuleList(
            DiTBlock(
                dim=dim,
                n_head=n_head,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(depth)
        )
        self.norm = AdaLN(dim)
        self.projection = nn.Linear(dim, mel_dim)
        self.long_skip_connection = (
            nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        )

    def forward(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, mel_dim] noisy mel-spectrogram
            m: [B, T, mel_dim] mixture mel-spectrogram
            c: [B, C, T_c, F] raw encoder output (T_c may differ from T)
            t: [B] diffusion timestep
            mask: Optional [B, T] boolean Tensor for self-attention (True = valid)

        Returns:
            out: [B, T, mel_dim]
        """
        seq_len = x.shape[1]
        t = self.time_emb(t)
        x = self.input_emb(x, m)
        c = self.cond_emb(c)
        rope = self.rotary_emb.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            x = block(x, t, c, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm(x, t)
        out = self.projection(x)

        return out
