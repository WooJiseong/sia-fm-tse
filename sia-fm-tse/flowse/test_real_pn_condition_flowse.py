import torch
import torch.nn as nn
import torchaudio

from model.cfm import CFM
from model.backbones.dit import DiT


class SpeakerConditionedCFM(nn.Module):
    def __init__(self, cfm, spk_dim=64, mel_dim=100):
        super().__init__()
        self.cfm = cfm
        self.spk_proj = nn.Linear(spk_dim, mel_dim)

    def wav_to_mel(self, wav):
        # wav: [B, samples]
        mel = self.cfm.mel_spec(wav)      # [B, mel, T]
        mel = mel.permute(0, 2, 1)        # [B, T, mel]
        return mel

    def forward(self, mix_wave, target_wave, text, cond_emb):
        # cond_emb: [B, 64, T, F]
        c_spk = cond_emb.mean(dim=(2, 3))     # [B, 64]
        spk = self.spk_proj(c_spk)            # [B, 100]
        spk = spk[:, None, :]                 # [B, 1, 100]

        noisy_mel = self.wav_to_mel(mix_wave)
        clean_mel = self.wav_to_mel(target_wave)

        # speaker-conditioned noisy mel
        noisy_mel = noisy_mel + spk

        return self.cfm(
            inp=noisy_mel,
            clean=clean_mel,
            text=text,
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

data = torch.load("../bridge_outputs/pn_condition_sample.pt", map_location="cpu")

mix_wave = data["mix_wave"].squeeze(1).to(device)       # [B, wav]
target_wave = data["target_wave"].squeeze(1).to(device) # [B, wav]
cond_emb = data["cond_emb"].to(device)                  # [B, 64, 751, 65]
sr = data["sample_rate"]

# FlowSE train config uses 24 kHz mel. Resample 16k -> 24k for this bridge test.
if sr != 24000:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000).to(device)
    mix_wave = resampler(mix_wave)
    target_wave = resampler(target_wave)

transformer = DiT(
    dim=256,
    depth=2,
    heads=4,
    dim_head=64,
    mel_dim=100,
    text_num_embeds=256,
).to(device)

cfm = CFM(
    transformer=transformer,
    num_channels=100,
).to(device)

model = SpeakerConditionedCFM(
    cfm=cfm,
    spk_dim=64,
    mel_dim=100,
).to(device)

text = [" "] * mix_wave.shape[0]

loss, cond, pred = model(
    mix_wave=mix_wave,
    target_wave=target_wave,
    text=text,
    cond_emb=cond_emb,
)

print("mix_wave:", mix_wave.shape)
print("target_wave:", target_wave.shape)
print("cond_emb:", cond_emb.shape)
print("cond:", cond.shape)
print("pred:", pred.shape)
print("loss:", loss.item())
