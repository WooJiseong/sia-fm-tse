"""Simple wrapper for pretrained pnenroll encoder"""

from pathlib import Path

import torch
import torch.nn as nn

from .model.GridnetAttnHead import GridNetBlock_attnhead
from .model.tfgridnet_crossattn_causal_single_emb import (
    TFGridNet_origcrossattn_causal_single_emb,
)
from .model.tfgridnet_encoder import TFGridNet_encoder
from .model.tfgridnet_KVfusion import TFGridNet_KVfusion


class Encoder(nn.Module):
    def __init__(self):
        """Simple wrapper for pretrained pnenroll encoder"""
        super().__init__()
        self.model = TFGridNet_origcrossattn_causal_single_emb(
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
            encoder=TFGridNet_encoder(
                num_ch=2,
                n_fft=128,
                stride=64,
                num_blocks=3,
                binaural=False,
            ),
            encoder_head=GridNetBlock_attnhead(
                layer_num=2,
                pooling_size=1,
                stride=1,
            ),
            train_encoder=False,
            train_encoder_head=False,
            fusion_layer=[0, 1],
            binaural=False,
        )

    def load(self, path: Path):
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()

    @torch.no_grad()
    def forward(
        self,
        pos: torch.Tensor,
        neg: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.model.encode(
            pos.sum(dim=1),
            neg.sum(dim=1),
        )
        return condition
