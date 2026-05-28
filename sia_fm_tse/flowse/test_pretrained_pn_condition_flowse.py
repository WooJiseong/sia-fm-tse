import torch
import torch.nn as nn
import torchaudio
import yaml

from sia_fm_tse.flowse.model import CFM, DiT


class SpeakerConditionedCFM(nn.Module):
    def __init__(self, cfm, spk_dim=64, mel_dim=100):
        super().__init__()
        self.cfm = cfm
        self.spk_proj = nn.Linear(spk_dim, mel_dim)

        # 중요: 처음에는 pretrained FlowSE를 망치지 않도록 0 초기화
        nn.init.zeros_(self.spk_proj.weight)
        nn.init.zeros_(self.spk_proj.bias)

    def wav_to_mel(self, wav):
        mel = self.cfm.mel_spec(wav)  # [B, mel, T]
        mel = mel.permute(0, 2, 1)  # [B, T, mel]
        return mel

    def forward(self, mix_wave, target_wave, text, cond_emb):
        c_spk = cond_emb.mean(dim=(2, 3))  # [B,64]
        spk = self.spk_proj(c_spk)  # [B,100]
        spk = spk[:, None, :]  # [B,1,100]

        noisy_mel = self.wav_to_mel(mix_wave)
        clean_mel = self.wav_to_mel(target_wave)

        # zero-init 상태에서는 처음엔 사실상 noisy_mel 그대로
        noisy_mel = noisy_mel + spk

        return self.cfm(
            inp=noisy_mel,
            clean=clean_mel,
            text=text,
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ===== load pretrained FlowSE config/checkpoint =====
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

model = SpeakerConditionedCFM(
    cfm=cfm,
    spk_dim=64,
    mel_dim=mel_dim,
).to(device)

model.eval()

print("loaded pretrained FlowSE")
print("checkpoint epoch:", ckpt.get("epoch"))
print("checkpoint best_loss:", ckpt.get("best_loss"))

# ===== load real PN condition sample =====
data = torch.load("../bridge_outputs/pn_condition_sample.pt", map_location="cpu")

mix_wave = data["mix_wave"].squeeze(1).to(device)  # [B,wav]
target_wave = data["target_wave"].squeeze(1).to(device)  # [B,wav]
cond_emb = data["cond_emb"].to(device)  # [B,64,751,65]
sr = data["sample_rate"]

# FlowSE mel spec is 24k
target_sr = mel_conf["target_sample_rate"]
if sr != target_sr:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr).to(
        device
    )
    mix_wave = resampler(mix_wave)
    target_wave = resampler(target_wave)

text = [" "] * mix_wave.shape[0]

with torch.no_grad():
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
print("loss:", float(loss))
