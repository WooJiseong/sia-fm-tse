from __future__ import annotations

import torch
from torch import nn


class TemporalAttentionPool(nn.Module):
    """Duration-independent pooling over PN enrollment frames."""

    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, 1),
        )

    def forward(self, cond_emb: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, freqs = cond_emb.shape
        x = cond_emb.permute(0, 3, 2, 1).reshape(batch * freqs, frames, channels)
        weight = torch.softmax(self.score(x), dim=1)
        pooled = (x * weight).sum(dim=1)
        return pooled.reshape(batch, freqs, channels)


class PNEmbeddingProjector(nn.Module):
    """Project PN encoder embeddings to frequency tokens for FlowSE.

    Input:  [B, C, T, F]
    Output: [B, F+1, out_dim] when add_global_token=True
    """

    def __init__(
        self,
        spk_dim: int = 64,
        freq_bins: int = 65,
        out_dim: int = 1024,
        temporal_pool: str = "mean_std",
        dropout: float = 0.0,
        add_global_token: bool = True,
    ):
        super().__init__()

        if temporal_pool not in {"mean", "mean_std", "attn"}:
            raise ValueError("temporal_pool must be one of: mean, mean_std, attn")

        self.spk_dim = spk_dim
        self.freq_bins = freq_bins
        self.temporal_pool = temporal_pool
        self.add_global_token = add_global_token

        pooled_dim = spk_dim * 2 if temporal_pool == "mean_std" else spk_dim

        self.attn_pool = TemporalAttentionPool(spk_dim) if temporal_pool == "attn" else None

        self.proj = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

        self.freq_pos = nn.Parameter(torch.zeros(freq_bins, out_dim))

        self.global_proj = (
            nn.Sequential(
                nn.LayerNorm(pooled_dim),
                nn.Linear(pooled_dim, out_dim),
            )
            if add_global_token
            else None
        )

    def pool_time(self, cond_emb: torch.Tensor) -> torch.Tensor:
        if cond_emb.ndim != 4:
            raise ValueError(f"expected [B, C, T, F], got {tuple(cond_emb.shape)}")

        if cond_emb.shape[1] != self.spk_dim:
            raise ValueError(f"expected C={self.spk_dim}, got C={cond_emb.shape[1]}")

        if cond_emb.shape[3] != self.freq_bins:
            raise ValueError(f"expected F={self.freq_bins}, got F={cond_emb.shape[3]}")

        if self.temporal_pool == "mean":
            return cond_emb.mean(dim=2).transpose(1, 2)

        if self.temporal_pool == "mean_std":
            mean = cond_emb.mean(dim=2)
            std = cond_emb.std(dim=2, unbiased=False)
            return torch.cat([mean, std], dim=1).transpose(1, 2)

        return self.attn_pool(cond_emb)

    def forward(self, cond_emb: torch.Tensor) -> torch.Tensor:
        pooled = self.pool_time(cond_emb)
        freq_tokens = self.proj(pooled) + self.freq_pos[None, :, :]

        if not self.add_global_token:
            return freq_tokens

        global_token = self.global_proj(pooled.mean(dim=1)).unsqueeze(1)
        return torch.cat([global_token, freq_tokens], dim=1)


def _infer_mel_dim(base_dit: nn.Module) -> int:
    if hasattr(base_dit, "mel_dim"):
        return int(base_dit.mel_dim)

    # FlowSE / F5-style DiT often has a final linear projection to mel_dim.
    for name in ["final_proj", "proj_out", "to_out", "to_pred"]:
        layer = getattr(base_dit, name, None)
        if isinstance(layer, nn.Linear):
            return int(layer.out_features)

    # Fallback: find the last Linear layer.
    last_linear = None
    for module in base_dit.modules():
        if isinstance(module, nn.Linear):
            last_linear = module

    if last_linear is not None:
        return int(last_linear.out_features)

    raise AttributeError("Could not infer mel_dim from base_dit")


class PNDiT(nn.Module):
    """Wrap pretrained DiT and inject PN tokens through cross-attention.

    cond_emb is not passed through CFM.forward().
    Instead, PNConditionedCFM temporarily stores cond_emb inside this wrapper.
    """

    def __init__(
        self,
        base_dit: nn.Module,
        spk_dim: int = 64,
        freq_bins: int = 65,
        temporal_pool: str = "mean_std",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.base_dit = base_dit
        self.current_cond_emb = None

        dim = getattr(base_dit, "dim", None)
        if dim is None:
            raise AttributeError("base_dit must have attribute `dim`")

        heads = getattr(base_dit, "heads", 8)
        mel_dim = _infer_mel_dim(base_dit)

        self.dim = dim
        self.mel_dim = mel_dim

        self.pn_projector = PNEmbeddingProjector(
            spk_dim=spk_dim,
            freq_bins=freq_bins,
            out_dim=dim,
            temporal_pool=temporal_pool,
            dropout=dropout,
            add_global_token=True,
        )

        self.query_proj = nn.Linear(mel_dim, dim)

        self.speaker_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True,
        )

        self.out_proj = nn.Linear(dim, mel_dim)

        # zero-init gate: 처음에는 pretrained FlowSE 출력을 거의 건드리지 않음
        self.speaker_attn_gate = nn.Parameter(torch.zeros(1))

    def set_condition(self, cond_emb: torch.Tensor):
        self.current_cond_emb = cond_emb

    def clear_condition(self):
        self.current_cond_emb = None

    def forward(self, *args, cond_emb: torch.Tensor | None = None, **kwargs):
        if cond_emb is None:
            cond_emb = self.current_cond_emb

        # 기존 DiT 출력: 보통 [B, T, mel_dim]
        x = self.base_dit(*args, **kwargs)

        if cond_emb is None:
            return x

        pn_tokens = self.pn_projector(cond_emb)

        # output-space x를 attention query dimension으로 투영
        query = self.query_proj(x)

        attn_out, _ = self.speaker_attn(
            query=query,
            key=pn_tokens,
            value=pn_tokens,
            need_weights=False,
        )

        delta = self.out_proj(attn_out)

        return x + self.speaker_attn_gate * delta


class PNConditionedCFM(nn.Module):
    """CFM wrapper that injects PN enrollment embedding without modifying CFM.forward()."""

    def __init__(self, cfm: nn.Module):
        super().__init__()
        self.cfm = cfm

    def forward(
        self,
        inp: torch.Tensor,
        clean: torch.Tensor,
        text,
        cond_emb: torch.Tensor,
    ):
        if not hasattr(self.cfm.transformer, "set_condition"):
            raise TypeError("cfm.transformer must be PNDiT and have set_condition()")

        self.cfm.transformer.set_condition(cond_emb)

        try:
            return self.cfm(
                inp=inp,
                clean=clean,
                text=text,
            )
        finally:
            self.cfm.transformer.clear_condition()
