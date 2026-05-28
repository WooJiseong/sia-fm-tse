import glob
import os

import torch
import torch.nn as nn
import torchaudio
import yaml
from torch.utils.data import DataLoader, Dataset

from sia_fm_tse.flowse.model import CFM, DiT


class PNCacheDataset(Dataset):
    def __init__(self, cache_dir):
        self.files = sorted(glob.glob(os.path.join(cache_dir, "sample_*.pt")))
        if len(self.files) == 0:
            raise RuntimeError(f"No cache files found in {cache_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = torch.load(self.files[idx], map_location="cpu")

        mix_wave = item["mix_wave"].squeeze(0).squeeze(0)  # [96000]
        target_wave = item["target_wave"].squeeze(0).squeeze(0)  # [96000]
        cond_emb = item["cond_emb"].squeeze(0)  # [64, 751, 65]
        sr = item["sample_rate"]

        return mix_wave, target_wave, cond_emb, sr


class PNAdapterFlowSE(nn.Module):
    def __init__(self, cfm, spk_dim=64, mel_dim=100):
        super().__init__()
        self.cfm = cfm
        self.spk_proj = nn.Linear(spk_dim, mel_dim)

        nn.init.zeros_(self.spk_proj.weight)
        nn.init.zeros_(self.spk_proj.bias)

    def forward(self, noisy_mel, clean_mel, text, cond_emb):
        c_spk = cond_emb.mean(dim=(2, 3))  # [B, 64]
        spk = self.spk_proj(c_spk)  # [B, 100]
        spk = spk[:, None, :]  # [B, 1, 100]

        speaker_cond_mel = noisy_mel + spk  # [B, N, 100]

        return self.cfm(
            inp=speaker_cond_mel,
            clean=clean_mel,
            text=text,
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "3"))
LR = float(os.environ.get("LR", "1e-3"))

cache_dir = "../bridge_outputs/pn_cache"

# ===== load pretrained FlowSE =====
ckpt_path = "wenetspeech4tts_Premium.pt.tar"
ckpt = torch.load(ckpt_path, map_location="cpu")
state = ckpt["model_state_dict"]

with open("config/train.yaml", "r") as f:
    conf = yaml.safe_load(f)

model_conf = conf["model"]
arch = dict(model_conf["arch"])
mel_conf = dict(model_conf["mel_spec"])

text_num_embeds = state["transformer.text_embed.text_embed.weight"].shape[0] - 1
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

cfm.load_state_dict(state, strict=True)
cfm = cfm.to(device)
cfm.eval()

for p in cfm.parameters():
    p.requires_grad = False

model = PNAdapterFlowSE(
    cfm=cfm,
    spk_dim=64,
    mel_dim=mel_dim,
).to(device)

optimizer = torch.optim.AdamW(model.spk_proj.parameters(), lr=LR)

dataset = PNCacheDataset(cache_dir)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=True,
)

print("loaded pretrained FlowSE")
print("checkpoint epoch:", ckpt.get("epoch"))
print("checkpoint best_loss:", ckpt.get("best_loss"))
print("cache samples:", len(dataset))
print("batch_size:", BATCH_SIZE)
print("epochs:", NUM_EPOCHS)

model.train()
cfm.eval()

global_step = 0
os.makedirs("output", exist_ok=True)

for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    count = 0

    for mix_wave, target_wave, cond_emb, sr in loader:
        global_step += 1

        mix_wave = mix_wave.to(device)
        target_wave = target_wave.to(device)
        cond_emb = cond_emb.to(device)

        # 현재 cache는 전부 16k라고 가정
        if int(sr[0]) != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=int(sr[0]),
                new_freq=target_sr,
            ).to(device)
            mix_wave = resampler(mix_wave)
            target_wave = resampler(target_wave)

        with torch.no_grad():
            noisy_mel = cfm.mel_spec(mix_wave).permute(0, 2, 1)
            clean_mel = cfm.mel_spec(target_wave).permute(0, 2, 1)

        text = [" "] * noisy_mel.shape[0]

        optimizer.zero_grad()

        loss, cond, pred = model(
            noisy_mel=noisy_mel,
            clean_mel=clean_mel,
            text=text,
            cond_emb=cond_emb,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.spk_proj.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        count += 1

        if global_step == 1 or global_step % 10 == 0:
            with torch.no_grad():
                spk_norm = model.spk_proj(cond_emb.mean(dim=(2, 3))).norm().item()
            print(
                f"epoch {epoch + 1:03d} | step {global_step:05d} | "
                f"loss {loss.item():.6f} | avg_loss {total_loss / count:.6f} | "
                f"spk_norm {spk_norm:.6f}"
            )

    epoch_loss = total_loss / max(count, 1)
    print(f"epoch {epoch + 1:03d} done | epoch_loss {epoch_loss:.6f}")

    torch.save(
        {
            "epoch": epoch + 1,
            "global_step": global_step,
            "spk_proj": model.spk_proj.state_dict(),
            "epoch_loss": epoch_loss,
        },
        f"output/pn_adapter_bs{BATCH_SIZE}_epoch{epoch + 1}.pt",
    )

print("dataset adapter training done")
