"""
Code for the Encoder Fusion Module
Adopted from the TFGridnet code provided in ESPnet: end-to-end speech processing toolkit and LookOnceToHear
- ESPnet: https://github.com/espnet/espnet
- LookOnceToHear: https://github.com/vb000/lookoncetohear
The modification includes the concatenation of the input two embedding sequences, and the addition of Segmentation Embeddings
"""

import math
from collections.abc import Sequence

import einops
import torch
import torch.nn as nn
from espnet2.enh.separator.tfgridnet_separator import GridNetBlock
from espnet2.torch_utils.get_layer_from_string import get_layer
from torch.nn.functional import softmax
from torch.nn.parameter import Parameter


class LayerNormalization4DCF(nn.Module):
    """
    Layer Normalization over channel (C) and frequency (F) axes of a [B, C, T, F] tensor.

    Standard nn.LayerNorm only normalizes over the last N dims, so we implement
    this manually to normalize over dim 1 (C) and dim 3 (F) simultaneously.
    """

    NORM_DIMS = (1, 3)  # C and F axes

    def __init__(self, n_channels: int, n_freqs: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Learnable scale/shift: [1, C, 1, F] — broadcastable over B and T
        param_shape = (1, n_channels, 1, n_freqs)
        self.gamma = Parameter(torch.ones(param_shape, dtype=torch.float32))
        self.beta = Parameter(torch.zeros(param_shape, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, F]
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D tensor [B, C, T, F], got {x.ndim}D")

        # Compute mean and std over C and F -> [B, 1, T, 1]
        mu = x.mean(dim=self.NORM_DIMS, keepdim=True)
        std = x.std(dim=self.NORM_DIMS, keepdim=True, unbiased=False)

        x_norm = (x - mu) / (std + self.eps)
        return x_norm * self.gamma + self.beta  # affine transform: [B, C, T, F]


class GridNetBlockAttnHead(nn.Module):
    """
    Fusion module that merges positive and negative conditioning embeddings
    via a stack of GridNetBlockAttn layers.

    Concatenates pos/neg along the time axis, injects segment embeddings to
    distinguish the two sources, then optionally trims back to pos length and
    runs additional refinement blocks.

    Args:
        layer_num:         Number of GridNetBlockAttn layers in the main stack.
        pooling_size:      (Reserved) pooling window size.
        stride:            (Reserved) pooling stride.
        return_clean_dvec: If True, projects the output to a 256-dim d-vector.
        out_dim:           If > 0, projects the output to this dimension instead.
        refine_layer_num:  Number of extra GridNetBlock layers appended after fusion.
        fusion_shortcut:   Layer indices that use residual addition instead of replacement.
        cut_pos:           If True, discards the neg portion of the output before refinement.

    Input:
        pos_cond: [B, C, T_pos, F]
        neg_cond: [B, C, T_neg, F]

    Output:
        x: [B, C, T_pos, F] if cut_pos else [B, C, T_pos + T_neg, F]
    """

    def __init__(
        self,
        layer_num: int,
        pooling_size: int,
        stride: int,
        return_clean_dvec: bool = False,
        out_dim: int = 0,
        refine_layer_num: int = 0,
        fusion_shortcut: Sequence[int] = (0,),
        cut_pos: bool = False,
    ):
        super().__init__()

        self.pooling_size = pooling_size
        self.stride = stride

        # Segment embedding: index 0 = pos, index 1 = neg.
        # Embeds into C*F space so it can be directly added to the [B, C, T, F] tensor.
        self.segment_embedding = nn.Embedding(2, 64 * 65)

        # Main fusion stack: each layer attends across the full concatenated time axis.
        self.model = nn.ModuleList(
            GridNetBlockAttn(
                emb_dim=64,
                n_freqs=65,
                n_head=4,
                eps=1e-5,
            )
            for _ in range(layer_num)
        )

        # Optional projection to a compact d-vector or custom output dim.
        if return_clean_dvec:
            self.embed_proj = nn.Sequential(
                nn.Linear(65 * 64, 256),
                nn.LayerNorm(256),
            )

        # Optional refinement stack applied after fusion (and after cut_pos if enabled).
        # Uses the original GridNetBlock (intra/inter RNN) rather than the attn variant.
        self.refine_layer_num = refine_layer_num
        if refine_layer_num > 0:
            self.pending_module = nn.ModuleList(
                GridNetBlock(
                    emb_dim=64,
                    emb_ks=1,
                    emb_hs=1,
                    n_freqs=65,
                    hidden_channels=64,
                    n_head=4,
                    approx_qk_dim=512,
                    activation="prelu",
                    eps=1.0e-5,
                )
                for _ in range(refine_layer_num)
            )

        self.fusion_shortcut = fusion_shortcut  # layer indices that use x += layer(x)
        self.cut_pos = cut_pos

        if out_dim != 0:
            assert not return_clean_dvec, (
                "hotfix for now: linear project for stylespeech is different from dvec output"
            )
            self.embed_proj = nn.Sequential(nn.Linear(65 * 64, out_dim))

    def forward(self, pos_cond: torch.Tensor, neg_cond: torch.Tensor):
        B, C, T_pos, F = pos_cond.shape
        _, _, T_neg, _ = neg_cond.shape

        # Concatenate along time: [B, C, T_pos + T_neg, F]
        x = torch.concat([pos_cond, neg_cond], dim=2)

        # Build per-frame segment indices: 0 for pos frames, 1 for neg frames -> [B, T_pos + T_neg]
        seg_idx = torch.concat(
            [
                torch.zeros((B, T_pos), device=pos_cond.device),
                torch.ones((B, T_neg), device=pos_cond.device),
            ],
            dim=1,
        ).to(torch.int32)

        # Embed segment indices and reshape to match x: [B, T, C*F] -> [B, C, T, F]
        seg_emb = self.segment_embedding(seg_idx)
        seg_emb = einops.rearrange(seg_emb, "b t (c f) -> b c t f", c=C, f=F)
        x += seg_emb

        # Main fusion layers; fusion_shortcut indices use residual addition.
        for i, layer in enumerate(self.model):
            if i in self.fusion_shortcut:
                x += layer(x)
            else:
                x = layer(x)

        # Discard neg portion if only pos output is needed.
        if self.cut_pos:
            x = x[:, :, :T_pos]

        # Optional refinement pass.
        if self.refine_layer_num > 0:
            for module in self.pending_module:
                x = module(x)

        return x


class GridNetBlockAttn(nn.Module):
    """
    Single attention block operating on [B, C, T, F] tensors.

    Performs multi-head self-attention along the time axis:
      1. Project x into Q, K, V per head via Conv2d.
      2. Flatten C and F dims -> treat each frame as a token.
      3. Compute scaled dot-product attention across T,
         scaled by E*F (the flattened Q/K dim) rather than C.
      4. Concatenate heads and project back to emb_dim.

    Args:
        emb_dim:       Channel dimension C of the input (= model width).
        n_freqs:       Frequency bins F.
        n_head:        Number of attention heads; emb_dim must be divisible by n_head.
        approx_qk_dim: Target total dim for Q/K before splitting into heads.
                       Each head gets E = ceil(approx_qk_dim / n_freqs) channels,
                       so the actual flattened Q/K dim per head is E * F.
        activation:    Activation layer name passed to get_layer().
        eps:           Epsilon for LayerNormalization4DCF.
    """

    def __init__(
        self,
        *,
        emb_dim: int,
        n_freqs: int,
        n_head: int = 4,
        approx_qk_dim: int = 512,
        activation: str = "prelu",
        eps: float = 1e-5,
    ):
        super().__init__()

        # Q/K projection dim per head: chosen so that n_head * E * n_freqs ≈ approx_qk_dim.
        E = math.ceil(approx_qk_dim / n_freqs)
        assert emb_dim % n_head == 0
        activation_layer: nn.Module = get_layer(activation)

        # Q projections: one Conv2d per head, [B, C, T, F] -> [B, E, T, F]
        self.attn_conv_Q = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(emb_dim, E, 1),
                get_layer(activation)(),
                LayerNormalization4DCF(n_channels=E, n_freqs=n_freqs, eps=eps),
            )
            for _ in range(n_head)
        )

        # K projections: same shape as Q, [B, C, T, F] -> [B, E, T, F]
        self.attn_conv_K = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(emb_dim, E, 1),
                activation_layer(),
                LayerNormalization4DCF(n_channels=E, n_freqs=n_freqs, eps=eps),
            )
            for _ in range(n_head)
        )

        # V projections: [B, C, T, F] -> [B, C/H, T, F]  (H := n_head)
        self.attn_conv_V = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim // n_head, 1),
                activation_layer(),
                LayerNormalization4DCF(
                    n_channels=emb_dim // n_head, n_freqs=n_freqs, eps=eps
                ),
            )
            for _ in range(n_head)
        )

        # Final projection after head concatenation: [B, C, T, F] -> [B, C, T, F]
        self.attn_concat_proj = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, 1),
            activation_layer(),
            LayerNormalization4DCF(n_channels=emb_dim, n_freqs=n_freqs, eps=eps),
        )

        self.n_head = n_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, F]
        _, _, _, F = x.shape

        # Project x into Q, K, V for each head
        all_Q = tuple(conv(x) for conv in self.attn_conv_Q)  # each: [B, E, T, F]
        all_K = tuple(conv(x) for conv in self.attn_conv_K)  # each: [B, E, T, F]
        all_V = tuple(conv(x) for conv in self.attn_conv_V)  # each: [B, C/H, T, F]

        # Stack heads along batch dim: [H*B, E or C/H, T, F]
        Q = torch.cat(all_Q, dim=0)
        K = torch.cat(all_K, dim=0)
        V = torch.cat(all_V, dim=0)

        # Flatten channel and freq into token dim: [H*B, T, E*F or (C/H)*F]
        Q = einops.rearrange(Q, "hb e t f -> hb t (e f)")
        K = einops.rearrange(K, "hb e t f -> hb t (e f)")
        V = einops.rearrange(V, "hb v t f -> hb t (v f)")  # V = C/H

        # Scaled dot-product attention over T, scaled by E*F (flattened Q/K dim)
        qk_dim = Q.shape[-1]  # E * F
        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (qk_dim**0.5)  # [H*B, T, T]
        attn_mat = softmax(attn_mat, dim=2)

        # Weighted sum of V: [H*B, T, (C/H)*F]
        attn_score = attn_mat @ V

        # Merge heads: [H*B, T, (C/H)*F] -> [B, C, T, F]
        out = einops.rearrange(attn_score, "(h b) t (v f) -> b (h v) t f",
                               f=F, h=self.n_head)  # fmt: skip

        # Final conv projection: [B, C, T, F]
        out = self.attn_concat_proj(out)
        return out
