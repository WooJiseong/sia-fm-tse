import torch

from model.backbones.dit import DiT
from model.cfm import CFM
from model.pn_conditioner import PNConditionedCFM, PNDiT


def test_pn_conditioned_cfm_shape():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dit = DiT(
        dim=128,
        depth=2,
        heads=4,
        dim_head=32,
        mel_dim=100,
        text_num_embeds=256,
    )

    cfm = CFM(
        transformer=base_dit,
        num_channels=100,
    )

    cfm.transformer = PNDiT(
        cfm.transformer,
        spk_dim=64,
        freq_bins=65,
        temporal_pool="mean_std",
    )

    model = PNConditionedCFM(cfm).to(device)

    batch, frames, mel_dim = 2, 64, 100

    noisy_mel = torch.randn(batch, frames, mel_dim, device=device)
    clean_mel = torch.randn(batch, frames, mel_dim, device=device)
    cond_emb = torch.randn(batch, 64, 123, 65, device=device)
    text = [" ", " "]

    loss, cond, pred = model(
        inp=noisy_mel,
        clean=clean_mel,
        text=text,
        cond_emb=cond_emb,
    )

    pn_tokens = model.cfm.transformer.pn_projector(cond_emb)

    assert loss.ndim == 0
    assert cond.shape == noisy_mel.shape
    assert pred.shape == clean_mel.shape
    assert pn_tokens.shape == (batch, 66, 128)
