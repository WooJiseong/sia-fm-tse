import itertools
import os
import sys

import torch
import torchaudio
import yaml

# ============================================================
# Import FlowSE modules first
# ============================================================
from flowse.model import CFM, DiT
from model.pn_conditioner import PNConditionedCFM, PNDiT
from torch.utils.data import DataLoader

# ============================================================
# Paths
# ============================================================

FLOWSE_ROOT = os.getcwd()
PN_REPO = os.environ.get(
    "PN_REPO",
    "/gpfs/home1/eri0529/SIA/paper/TSE-through-Positive-Negative-Enroll",
)

FLOWSE_CKPT = os.environ.get("FLOWSE_CKPT", "wenetspeech4tts_Premium.pt.tar")
PN_CKPT = os.environ.get(
    "PN_CKPT", os.path.join(PN_REPO, "output/proposed-monaural.pt")
)

SAVE_DIR = os.environ.get("SAVE_DIR", "output")

# ============================================================
# Import PN-TSE modules despite same package name "model"
# ============================================================


def import_pn_modules(pn_repo: str):
    import importlib

    pn_repo = os.path.abspath(pn_repo)
    flowse_root = os.path.abspath(FLOWSE_ROOT)

    saved_modules = {
        k: v
        for k, v in sys.modules.items()
        if k == "model"
        or k.startswith("model.")
        or k == "dataset"
        or k.startswith("dataset.")
    }

    for k in list(sys.modules.keys()):
        if (
            k == "model"
            or k.startswith("model.")
            or k == "dataset"
            or k.startswith("dataset.")
        ):
            del sys.modules[k]

    saved_path = list(sys.path)

    # flowse 경로와 현재 경로가 PN import를 가로채지 못하게 제거
    sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != flowse_root]

    # PN repo를 최우선으로 둠
    sys.path.insert(0, pn_repo)

    try:
        LibriDataset_single_emb = importlib.import_module(
            "dataset.LibriSpeech_single_emb"
        ).LibriDataset_single_emb

        TFGridNet_encoder = importlib.import_module(
            "model.tfgridnet_encoder"
        ).TFGridNet_encoder

        GridNetBlock_attnhead = importlib.import_module(
            "model.GridnetAttnHead"
        ).GridNetBlock_attnhead

        TFGridNet_KVfusion = importlib.import_module(
            "model.tfgridnet_KVfusion"
        ).TFGridNet_KVfusion

        TFGridNet_origcrossattn_causal_single_emb = importlib.import_module(
            "model.tfgridnet_crossattn_causal_single_emb"
        ).TFGridNet_origcrossattn_causal_single_emb

    finally:
        for k in list(sys.modules.keys()):
            if (
                k == "model"
                or k.startswith("model.")
                or k == "dataset"
                or k.startswith("dataset.")
            ):
                del sys.modules[k]

        sys.path = saved_path
        sys.modules.update(saved_modules)

    return (
        LibriDataset_single_emb,
        TFGridNet_encoder,
        GridNetBlock_attnhead,
        TFGridNet_KVfusion,
        TFGridNet_origcrossattn_causal_single_emb,
    )


(
    LibriDataset_single_emb,
    TFGridNet_encoder,
    GridNetBlock_attnhead,
    TFGridNet_KVfusion,
    TFGridNet_origcrossattn_causal_single_emb,
) = import_pn_modules(PN_REPO)


# ============================================================
# Config
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
print("PN_REPO:", PN_REPO)
print("FLOWSE_CKPT:", FLOWSE_CKPT)
print("PN_CKPT:", PN_CKPT)

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "10"))
STEPS_PER_EPOCH = int(os.environ.get("STEPS_PER_EPOCH", "500"))
LR = float(os.environ.get("LR", "1e-4"))
TEMPORAL_POOL = os.environ.get("TEMPORAL_POOL", "mean_std")
PN_DROPOUT = float(os.environ.get("PN_DROPOUT", "0.0"))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "1.0"))

sample_rate = 16000
source_num = 3
enroll_num = 3
active_num = [-1, 1, -1]
wave_length = 3
pos_example_length = 48000
neg_example_length = 48000
snr_db_range = [0, 0]
filling_pattern = "repeat"
dvec_rate = 50
reverb_cond = False

test_data_dir = os.path.join(PN_REPO, "data/LibriSpeech/LibriSpeech/_test_data/")
noise_dir = os.path.join(PN_REPO, "data/wham_noise/")


# ============================================================
# Build frozen PN-TSE encoder model
# ============================================================

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

pn_model = TFGridNet_origcrossattn_causal_single_emb(
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

pn_ckpt = torch.load(PN_CKPT, map_location="cpu")
pn_model.load_state_dict(pn_ckpt["state_dict"], strict=True)
pn_model.eval()

for p in pn_model.parameters():
    p.requires_grad = False

print("loaded frozen PN-TSE encoder")


# ============================================================
# Build online PN dataset
# reproducable=False follows the training-style random sampling behavior.
# ============================================================

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
    reproducable=False,
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

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=True,
)

loader_iter = itertools.cycle(loader)

print("online dataset length:", len(dataset))
print("batch_size:", BATCH_SIZE)
print("epochs:", NUM_EPOCHS)
print("steps_per_epoch:", STEPS_PER_EPOCH)
print("lr:", LR)


# ============================================================
# Build pretrained FlowSE + PN adapter
# ============================================================

flowse_ckpt = torch.load(FLOWSE_CKPT, map_location="cpu")
flowse_state = flowse_ckpt["model_state_dict"]

with open("config/train.yaml") as f:
    conf = yaml.safe_load(f)

model_conf = conf["model"]
arch = dict(model_conf["arch"])
mel_conf = dict(model_conf["mel_spec"])

text_num_embeds = flowse_state["transformer.text_embed.text_embed.weight"].shape[0] - 1
mel_dim = mel_conf["n_mel_channels"]
target_sr = mel_conf["target_sample_rate"]

transformer = DiT(
    **arch,
    text_num_embeds=text_num_embeds,
    mel_dim=mel_dim,
)

cfm = CFM(
    transformer=transformer,
    audio_drop_prob=model_conf.get("audio_drop_prob", 0.0),
    cond_drop_prob=model_conf.get("cond_drop_prob", 0.0),
    num_channels=mel_dim,
    mel_spec_kwargs=mel_conf,
)

cfm.load_state_dict(flowse_state, strict=True)

cfm.transformer = PNDiT(
    cfm.transformer,
    spk_dim=64,
    freq_bins=65,
    temporal_pool=TEMPORAL_POOL,
    dropout=PN_DROPOUT,
)

cfm = cfm.to(device)
cfm.eval()

for p in cfm.parameters():
    p.requires_grad = False

for p in cfm.transformer.pn_projector.parameters():
    p.requires_grad = True

for p in cfm.transformer.speaker_attn.parameters():
    p.requires_grad = True

for p in cfm.transformer.query_proj.parameters():
    p.requires_grad = True

for p in cfm.transformer.out_proj.parameters():
    p.requires_grad = True

cfm.transformer.speaker_attn_gate.requires_grad = True

model = PNConditionedCFM(cfm=cfm).to(device)

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=LR)

print("loaded pretrained FlowSE")
print("checkpoint epoch:", flowse_ckpt.get("epoch"))
print("checkpoint best_loss:", flowse_ckpt.get("best_loss"))
print("trainable params:", sum(p.numel() for p in trainable_params))


# ============================================================
# Train
# ============================================================

os.makedirs(SAVE_DIR, exist_ok=True)

global_step = 0

model.train()
cfm.eval()
cfm.transformer.pn_projector.train()
cfm.transformer.speaker_attn.train()
cfm.transformer.query_proj.train()
cfm.transformer.out_proj.train()

for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    count = 0

    for step in range(STEPS_PER_EPOCH):
        global_step += 1

        audio, pos, neg = next(loader_iter)

        audio = audio.to(device)  # [B, source/noise, ch, wav]
        pos = pos.to(device)
        neg = neg.to(device)

        mix_wave = audio.sum(dim=1).squeeze(1)  # [B, wav]
        target_wave = audio[:, : active_num[1]].sum(dim=1).squeeze(1)  # [B, wav]

        with torch.no_grad():
            # Dataset already gives 3s pos/neg according to pos_example_length/neg_example_length.
            # Because reproducable=False, repeated calls generate different enrollment examples.
            cond_emb = pn_model.encode(
                pos.sum(dim=1),
                neg.sum(dim=1),
            )  # [B, 64, T, 65]

            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=target_sr,
                ).to(device)
                mix_wave = resampler(mix_wave)
                target_wave = resampler(target_wave)

            noisy_mel = cfm.mel_spec(mix_wave).permute(0, 2, 1)
            clean_mel = cfm.mel_spec(target_wave).permute(0, 2, 1)

        text = [" "] * noisy_mel.shape[0]

        optimizer.zero_grad()

        loss, cond, pred = model(
            inp=noisy_mel,
            clean=clean_mel,
            text=text,
            cond_emb=cond_emb,
        )

        if not torch.isfinite(loss):
            print(
                f"non-finite loss at epoch {epoch + 1}, step {step + 1}, global_step {global_step}: {loss.item()}"
            )
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        count += 1

        if global_step == 1 or global_step % 20 == 0:
            with torch.no_grad():
                pn_tokens = model.cfm.transformer.pn_projector(cond_emb)
                pn_norm = pn_tokens.norm(dim=-1).mean().item()
                gate = model.cfm.transformer.speaker_attn_gate.item()

            print(
                f"epoch {epoch + 1:03d} | step {global_step:06d} | "
                f"loss {loss.item():.6f} | avg_loss {total_loss / max(count, 1):.6f} | "
                f"pn_token_norm {pn_norm:.6f} | gate {gate:.6f}",
                flush=True,
            )

    epoch_loss = total_loss / max(count, 1)
    print(f"epoch {epoch + 1:03d} done | epoch_loss {epoch_loss:.6f}", flush=True)

    save_path = os.path.join(
        SAVE_DIR,
        f"pn_online_adapter_bs{BATCH_SIZE}_epoch{epoch + 1}.pt",
    )

    torch.save(
        {
            "epoch": epoch + 1,
            "global_step": global_step,
            "pn_projector": model.cfm.transformer.pn_projector.state_dict(),
            "speaker_attn": model.cfm.transformer.speaker_attn.state_dict(),
            "query_proj": model.cfm.transformer.query_proj.state_dict(),
            "out_proj": model.cfm.transformer.out_proj.state_dict(),
            "speaker_attn_gate": model.cfm.transformer.speaker_attn_gate.detach().cpu(),
            "epoch_loss": epoch_loss,
            "temporal_pool": TEMPORAL_POOL,
            "batch_size": BATCH_SIZE,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "lr": LR,
            "online_pn_encoder": True,
            "pn_repo": PN_REPO,
        },
        save_path,
    )

    print("saved:", save_path, flush=True)

print("online PN-conditioned FlowSE adapter training done")
