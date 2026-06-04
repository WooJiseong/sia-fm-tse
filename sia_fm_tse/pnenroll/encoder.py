from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .model import (
    GridNetBlock_attnhead,
    TFGridNet_KVfusion,
    TFGridNet_encoder,
    TFGridNet_origcrossattn_causal_single_emb,
)


def build_pn_encoder() -> TFGridNet_origcrossattn_causal_single_emb:
    # Keep these hyperparameters aligned with proposed-monaural.pt.
    encoder = TFGridNet_encoder(
        num_ch=2,
        n_fft=128,
        stride=64,
        num_blocks=3,
        binaural=False,
    )

    encoder_head = GridNetBlock_attnhead(
        layer_num=2,
        pooling_size=1,
        stride=1,
    )

    return TFGridNet_origcrossattn_causal_single_emb(
        n_fft=128,
        stride=64,
        n_layers=3,
        lstm_hidden_units=64,
        emb_dim=64,
        emb_ks=1,
        model_normalize=True,
        Fusion_class=TFGridNet_KVfusion,
        pooling_size=40,
        fusion_stride=40,
        encoder=encoder,
        encoder_head=encoder_head,
        train_encoder=False,
        train_encoder_head=False,
        fusion_layer=[0, 1],
        binaural=False,
    )


def load_frozen_pn_encoder(
    checkpoint_path: str | Path,
    device: torch.device | str,
    *,
    strict: bool = True,
) -> nn.Module:
    model = build_pn_encoder().to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    # FlowSE conditioning only calls model.encode(), so load the PN condition
    # encoder strictly and ignore extraction-branch weights that are never used.
    model_state = model.state_dict()
    condition_prefixes = ("encoder.", "encoder_head.")
    condition_state = {}
    mismatched = []

    for key, value in state_dict.items():
        if not key.startswith(condition_prefixes):
            continue
        if key not in model_state:
            mismatched.append(f"{key}: not present in current model")
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            mismatched.append(
                f"{key}: checkpoint {tuple(value.shape)} != model {tuple(model_state[key].shape)}"
            )
            continue
        condition_state[key] = value

    missing = [
        key
        for key in model_state
        if key.startswith(condition_prefixes) and key not in condition_state
    ]
    if strict and (missing or mismatched):
        details = []
        if missing:
            details.append("missing condition keys:\n  " + "\n  ".join(missing))
        if mismatched:
            details.append("mismatched condition keys:\n  " + "\n  ".join(mismatched))
        raise RuntimeError("\n".join(details))

    model.load_state_dict(condition_state, strict=False)
    model.eval()

    # The PN model acts as a frozen condition encoder in this training path.
    for param in model.parameters():
        param.requires_grad = False

    return model
