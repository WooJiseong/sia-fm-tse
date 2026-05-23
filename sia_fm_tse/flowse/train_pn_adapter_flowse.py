import torch
import torch.nn as nn
import torchaudio
import yaml

from model import DiT, CFM


class PNAdapterFlowSE(nn.Module):
    def __init__(self, cfm, spk_dim=64, mel_dim=100):
        super().__init__()
        self.cfm = cfm
        self.spk_proj = nn.Linear(spk_dim, mel_dim)

        # pretrained FlowSE를 처음부터 망치지 않도록 0 초기화
        nn.init.zeros_(self.spk_proj.weight)
        nn.init.zeros_(self.spk_proj.bias)

    def forward(self, noisy_mel, clean_mel, text, cond_emb):
        # cond_emb: [B, 64, T, F]
        c_spk = cond_emb.mean(dim=(2, 3))     # [B, 64]
        spk = self.spk_proj(c_spk)            # [B, 100]
        spk = spk[:, None, :]                 # [B, 1, 100]

        speaker_cond_mel = noisy_mel + spk    # [B, N, 100]

        return self.cfm(
            inp=speaker_cond_mel,
            clean=clean_mel,
            text=text,
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

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

# FlowSE는 freeze
for p in cfm.parameters():
    p.requires_grad = False

model = PNAdapterFlowSE(
    cfm=cfm,
    spk_dim=64,
    mel_dim=mel_dim,
).to(device)

# spk_proj만 학습
optimizer = torch.optim.AdamW(model.spk_proj.parameters(), lr=1e-3)

print("loaded pretrained FlowSE")
print("checkpoint epoch:", ckpt.get("epoch"))
print("checkpoint best_loss:", ckpt.get("best_loss"))

# ===== load bridge sample =====
data = torch.load("../bridge_outputs/pn_condition_sample.pt", map_location="cpu")

mix_wave = data["mix_wave"].squeeze(1).to(device)       # [B, wav]
target_wave = data["target_wave"].squeeze(1).to(device) # [B, wav]
cond_emb = data["cond_emb"].to(device)                  # [B, 64, 751, 65]
sr = data["sample_rate"]

target_sr = mel_conf["target_sample_rate"]
if sr != target_sr:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr).to(device)
    mix_wave = resampler(mix_wave)
    target_wave = resampler(target_wave)

# mel은 미리 계산
with torch.no_grad():
    noisy_mel = cfm.mel_spec(mix_wave).permute(0, 2, 1)    # [B, N, 100]
    clean_mel = cfm.mel_spec(target_wave).permute(0, 2, 1) # [B, N, 100]

text = [" "] * noisy_mel.shape[0]

print("mix_wave:", mix_wave.shape)
print("target_wave:", target_wave.shape)
print("noisy_mel:", noisy_mel.shape)
print("clean_mel:", clean_mel.shape)
print("cond_emb:", cond_emb.shape)

# ===== adapter sanity training =====
model.train()
cfm.eval()

num_steps = 20

for step in range(num_steps):
    optimizer.zero_grad()

    # sanity check용: Flow Matching 내부 random time/noise를 고정해서 loss 감소를 보기 쉽게 함
    torch.manual_seed(1234)

    loss, cond, pred = model(
        noisy_mel=noisy_mel,
        clean_mel=clean_mel,
        text=text,
        cond_emb=cond_emb,
    )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.spk_proj.parameters(), 1.0)
    optimizer.step()

    if step == 0 or (step + 1) % 5 == 0:
        with torch.no_grad():
            spk_norm = model.spk_proj(cond_emb.mean(dim=(2, 3))).norm().item()
        print(f"step {step+1:03d} | loss {loss.item():.6f} | spk_norm {spk_norm:.6f}")

print("adapter training sanity check done")
