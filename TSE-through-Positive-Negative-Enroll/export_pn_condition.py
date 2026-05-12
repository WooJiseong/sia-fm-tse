import os
import torch
import random

from dataset.LibriSpeech_single_emb import LibriDataset_single_emb
from model.tfgridnet_encoder import TFGridNet_encoder
from model.GridnetAttnHead import GridNetBlock_attnhead
from model.tfgridnet_KVfusion import TFGridNet_KVfusion
from model.tfgridnet_crossattn_causal_single_emb import TFGridNet_origcrossattn_causal_single_emb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ===== same basic setting as eval_monaural.py =====
test_data_dir = "data/LibriSpeech/LibriSpeech/_test_data/"
noise_dir = "data/wham_noise/"

source_num = 3
enroll_num = 3
active_num = [-1, 1, -1]
sample_rate = 16000
wave_length = 6
pos_example_length = 48000
neg_example_length = 48000
snr_db_range = [-2.5, 2.5]
filling_pattern = "repeat"
dvec_rate = 50
reverb_cond = False

# ===== build PN-TSE model =====
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

model = TFGridNet_origcrossattn_causal_single_emb(
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
).to(device)

ckpt = torch.load("output/proposed-monaural.pt", map_location="cpu")
model.load_state_dict(ckpt["state_dict"], strict=True)
model.eval()

# ===== dataset =====
dataset = LibriDataset_single_emb(
    test_data_dir,
    sample_rate=sample_rate,
    wave_length=wave_length * sample_rate,
    pos_example_length=pos_example_length,
    neg_example_length=neg_example_length,
    snr_db_range=snr_db_range,
    source_num=source_num,
    min_source_num=source_num,
    enroll_num=enroll_num,
    min_enroll_num=enroll_num,
    active_num=active_num,
    reproducable=True,
    normalize=False,
    filling_pattern=filling_pattern,
    return_dvec=False,
    dvec_rate=dvec_rate,
    include_silent=False,
    special_spk=[],
    reverb="none",
    binaural=False,
    reverb_cond=reverb_cond,
    zero_in_tgt=False,
    noise_dir=noise_dir + "tt/",
    same_disturb=False,
)

with torch.no_grad():
    audio, pos, neg = dataset[0]

    audio = audio.to(device)[None]  # [B, source/noise, ch, wav]
    pos = pos.to(device)[None]
    neg = neg.to(device)[None]

    mix_wave = audio.sum(dim=1)  # [B, ch, wav]
    target_wave = audio[:, :active_num[1]].sum(dim=1)  # [B, ch, wav]

    cond_emb = model.encode(pos.sum(dim=1), neg.sum(dim=1))  # [B, 64, T, 65]

print("audio:", audio.shape)
print("mix_wave:", mix_wave.shape)
print("target_wave:", target_wave.shape)
print("pos:", pos.shape)
print("neg:", neg.shape)
print("cond_emb:", cond_emb.shape)

save_dir = "../bridge_outputs"
os.makedirs(save_dir, exist_ok=True)

torch.save(
    {
        "mix_wave": mix_wave.detach().cpu(),
        "target_wave": target_wave.detach().cpu(),
        "pos": pos.detach().cpu(),
        "neg": neg.detach().cpu(),
        "cond_emb": cond_emb.detach().cpu(),
        "sample_rate": sample_rate,
    },
    os.path.join(save_dir, "pn_condition_sample.pt"),
)

print("saved:", os.path.join(save_dir, "pn_condition_sample.pt"))
