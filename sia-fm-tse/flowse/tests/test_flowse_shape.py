import torch
from model.cfm import CFM
from model.backbones.dit import DiT

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

model = CFM(
    transformer=transformer,
    num_channels=100,
).to(device)

B, N, D = 2, 200, 100
noisy = torch.randn(B, N, D, device=device)
clean = torch.randn(B, N, D, device=device)
text = [" ", " "]

loss, cond, pred = model(inp=noisy, clean=clean, text=text)

print("loss:", loss.item())
print("noisy:", noisy.shape)
print("clean:", clean.shape)
print("cond:", cond.shape)
print("pred:", pred.shape)
