import torch
import torch.nn as nn

from model.cfm import CFM
from model.backbones.dit import DiT


class SpeakerConditionedCFM(nn.Module):
    def __init__(self, cfm, spk_dim=64, mel_dim=100):
        super().__init__()
        self.cfm = cfm
        self.spk_proj = nn.Linear(spk_dim, mel_dim)

    def forward(self, inp, clean, text, cond_emb):
        # cond_emb: [B, 64, T, F]
        c_spk = cond_emb.mean(dim=(2, 3))   # [B, 64]
        spk = self.spk_proj(c_spk)          # [B, 100]
        spk = spk[:, None, :]               # [B, 1, 100]

        # inp: noisy mel condition [B, N, 100]
        speaker_conditioned_inp = inp + spk # broadcast to [B, N, 100]

        return self.cfm(
            inp=speaker_conditioned_inp,
            clean=clean,
            text=text,
        )


device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

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

B, N, D = 2, 200, 100

noisy_mel = torch.randn(B, N, D, device=device)
clean_mel = torch.randn(B, N, D, device=device)

# PN-TSE encoder에서 나온 cond_emb shape와 동일하게 dummy 생성
cond_emb = torch.randn(B, 64, 751, 65, device=device)

text = [" ", " "]

loss, cond, pred = model(
    inp=noisy_mel,
    clean=clean_mel,
    text=text,
    cond_emb=cond_emb,
)

print("loss:", loss.item())
print("noisy_mel:", noisy_mel.shape)
print("clean_mel:", clean_mel.shape)
print("cond_emb:", cond_emb.shape)
print("cond:", cond.shape)
print("pred:", pred.shape)
