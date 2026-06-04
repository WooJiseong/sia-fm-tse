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



class PNFull2DProjector(nn.Module):
    """Project full time-frequency PN embedding into DiT token space.

    Input:
        cond_emb: [B, C, T, F]

    Output:
        tokens: [B, 1 + T*F, D] if add_global_token=True
                [B, T*F, D] otherwise
    """

    def __init__(
        self,
        spk_dim: int = 64,
        freq_bins: int = 65,
        out_dim: int = 1024,
        dropout: float = 0.0,
        add_global_token: bool = True,
        max_tokens: int | None = None,
    ):
        super().__init__()

        self.spk_dim = spk_dim
        self.freq_bins = freq_bins
        self.out_dim = out_dim
        self.add_global_token = add_global_token
        self.max_tokens = max_tokens

        self.norm = nn.LayerNorm(spk_dim)

        self.proj = nn.Sequential(
            nn.Linear(spk_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

        # Frequency identity is preserved; time order is preserved by token order.
        self.freq_pos = nn.Parameter(torch.zeros(freq_bins, out_dim))

        self.global_proj = (
            nn.Sequential(
                nn.LayerNorm(spk_dim),
                nn.Linear(spk_dim, out_dim),
            )
            if add_global_token
            else None
        )

    def forward(self, cond_emb: torch.Tensor) -> torch.Tensor:
        if cond_emb.ndim != 4:
            raise ValueError(f"expected [B, C, T, F], got {tuple(cond_emb.shape)}")

        B, C, T, F = cond_emb.shape

        if C != self.spk_dim:
            raise ValueError(f"expected C={self.spk_dim}, got C={C}")

        if F != self.freq_bins:
            raise ValueError(f"expected F={self.freq_bins}, got F={F}")

        # [B, C, T, F] -> [B, T, F, C]
        x = cond_emb.permute(0, 2, 3, 1).contiguous()

        # [B, T, F, C] -> [B, T, F, D]
        x = self.norm(x)
        tokens = self.proj(x)
        tokens = tokens + self.freq_pos[None, None, :, :]

        # [B, T, F, D] -> [B, T*F, D]
        tokens = tokens.view(B, T * F, self.out_dim)

        # Optional emergency token cap. None means full 2D tokens are preserved.
        if self.max_tokens is not None and tokens.shape[1] > self.max_tokens:
            idx = torch.linspace(
                0,
                tokens.shape[1] - 1,
                steps=self.max_tokens,
                device=tokens.device,
            ).long()
            tokens = tokens.index_select(dim=1, index=idx)

        if not self.add_global_token:
            return tokens

        # Global summary token from full time-frequency embedding.
        global_feat = cond_emb.mean(dim=(2, 3))  # [B, C]
        global_token = self.global_proj(global_feat).unsqueeze(1)

        return torch.cat([global_token, tokens], dim=1)


class PNTimeProjector(nn.Module):
    """Project PN embedding to time-axis tokens for cross-attention.

    Input:
        cond_emb: [B, C, T, F]

    Output:
        tokens: [B, 1 + T, D] if add_global_token=True
                [B, T, D] otherwise
    """

    def __init__(
        self,
        spk_dim: int = 64,
        freq_bins: int = 65,
        out_dim: int = 1024,
        dropout: float = 0.0,
        add_global_token: bool = True,
        max_tokens: int | None = None,
    ):
        super().__init__()

        self.spk_dim = spk_dim
        self.freq_bins = freq_bins
        self.out_dim = out_dim
        self.add_global_token = add_global_token
        self.max_tokens = max_tokens

        self.norm = nn.LayerNorm(spk_dim * freq_bins)
        self.proj = nn.Sequential(
            nn.Linear(spk_dim * freq_bins, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.global_proj = (
            nn.Sequential(
                nn.LayerNorm(spk_dim),
                nn.Linear(spk_dim, out_dim),
            )
            if add_global_token
            else None
        )

    def forward(self, cond_emb: torch.Tensor) -> torch.Tensor:
        if cond_emb.ndim != 4:
            raise ValueError(f"expected [B, C, T, F], got {tuple(cond_emb.shape)}")

        B, C, T, F = cond_emb.shape
        if C != self.spk_dim:
            raise ValueError(f"expected C={self.spk_dim}, got C={C}")
        if F != self.freq_bins:
            raise ValueError(f"expected F={self.freq_bins}, got F={F}")

        tokens = cond_emb.permute(0, 2, 1, 3).reshape(B, T, C * F)
        tokens = self.proj(self.norm(tokens))

        if self.max_tokens is not None and tokens.shape[1] > self.max_tokens:
            idx = torch.linspace(
                0,
                tokens.shape[1] - 1,
                steps=self.max_tokens,
                device=tokens.device,
            ).long()
            tokens = tokens.index_select(dim=1, index=idx)

        if not self.add_global_token:
            return tokens

        global_feat = cond_emb.mean(dim=(2, 3))
        global_token = self.global_proj(global_feat).unsqueeze(1)
        return torch.cat([global_token, tokens], dim=1)


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
        token_mode: str = "freq_pool",
        max_full_tokens: int | None = None,
        injection_mode: str = "output",
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

        self.token_mode = token_mode
        self.injection_mode = injection_mode

        if injection_mode not in {"output", "block"}:
            raise ValueError(f"unknown injection_mode: {injection_mode}")

        # Different PN tokenizations let us compare frequency pooling, full TF tokens,
        # and T-axis attention without touching the training loop.
        if token_mode == "freq_pool":
            self.pn_projector = PNEmbeddingProjector(
                spk_dim=spk_dim,
                freq_bins=freq_bins,
                out_dim=dim,
                temporal_pool=temporal_pool,
                dropout=dropout,
                add_global_token=True,
            )
        elif token_mode == "full_2d":
            self.pn_projector = PNFull2DProjector(
                spk_dim=spk_dim,
                freq_bins=freq_bins,
                out_dim=dim,
                dropout=dropout,
                add_global_token=True,
                max_tokens=max_full_tokens,
            )
        elif token_mode == "time_flatten":
            self.pn_projector = PNTimeProjector(
                spk_dim=spk_dim,
                freq_bins=freq_bins,
                out_dim=dim,
                dropout=dropout,
                add_global_token=True,
                max_tokens=max_full_tokens,
            )
        else:
            raise ValueError(f"unknown token_mode: {token_mode}")

        self.query_proj = nn.Linear(mel_dim, dim)

        self.speaker_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True,
        )

        self.out_proj = nn.Linear(dim, mel_dim)

        # Start with a tiny non-zero residual so gradients reach the adapter.
        self.speaker_attn_gate = nn.Parameter(torch.full((1,), 1e-2))

    def set_condition(self, cond_emb: torch.Tensor):
        self.current_cond_emb = cond_emb

    def clear_condition(self):
        self.current_cond_emb = None

    def _apply_pn_hidden(self, hidden: torch.Tensor, pn_tokens: torch.Tensor) -> torch.Tensor:
        """Inject PN tokens into hidden DiT states: hidden [B, T, D]."""
        attn_out, _ = self.speaker_attn(
            query=hidden,
            key=pn_tokens,
            value=pn_tokens,
            need_weights=False,
        )
        return hidden + self.speaker_attn_gate * attn_out

    def _forward_block_injection(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        text: torch.Tensor,
        time: torch.Tensor,
        drop_audio_cond,
        drop_text,
        mask=None,
        pn_tokens: torch.Tensor | None = None,
    ):
        """DiT forward with PN attention injected after every DiT block."""
        base = self.base_dit

        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        t = base.time_embed(time)
        text_embed = base.text_embed(text, seq_len, drop_text=drop_text)
        x = base.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)

        rope = base.rotary_embed.forward_from_seq_len(seq_len)

        if base.long_skip_connection is not None:
            residual = x

        for block in base.transformer_blocks:
            if base.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(base.ckpt_wrapper(block), x, t, mask, rope)
            else:
                x = block(x, t, mask=mask, rope=rope)

            # This is the stronger conditioning path: PN tokens shape every block.
            if pn_tokens is not None:
                x = self._apply_pn_hidden(x, pn_tokens)

        if base.long_skip_connection is not None:
            x = base.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = base.norm_out(x, t)
        output = base.proj_out(x)
        return output

    def forward(self, *args, cond_emb: torch.Tensor | None = None, **kwargs):
        if cond_emb is None:
            cond_emb = self.current_cond_emb

        if cond_emb is None:
            return self.base_dit(*args, **kwargs)

        pn_tokens = self.pn_projector(cond_emb)

        if self.injection_mode == "block":
            return self._forward_block_injection(
                x=kwargs["x"],
                cond=kwargs["cond"],
                text=kwargs["text"],
                time=kwargs["time"],
                drop_audio_cond=kwargs["drop_audio_cond"],
                drop_text=kwargs["drop_text"],
                mask=kwargs.get("mask", None),
                pn_tokens=pn_tokens,
            )

        # Lighter alternative: apply PN residual only to the predicted vector field.
        x = self.base_dit(*args, **kwargs)

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
