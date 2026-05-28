import torch
import yaml

from sia_fm_tse.flowse.model import CFM, DiT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

ckpt_path = "wenetspeech4tts_Premium.pt.tar"
ckpt = torch.load(ckpt_path, map_location="cpu")
state = ckpt["model_state_dict"]

print("ckpt keys:", ckpt.keys())
print("epoch:", ckpt.get("epoch"))
print("best_loss:", ckpt.get("best_loss"))

with open("config/train.yaml", "r") as f:
    conf = yaml.safe_load(f)

model_conf = conf["model"]
arch = dict(model_conf["arch"])
mel_conf = dict(model_conf["mel_spec"])

# text embedding 크기는 checkpoint에서 직접 추론
text_embed_weight = state["transformer.text_embed.text_embed.weight"]
text_num_embeds = text_embed_weight.shape[0] - 1

mel_dim = mel_conf["n_mel_channels"]

print("text_num_embeds:", text_num_embeds)
print("mel_dim:", mel_dim)
print("arch:", arch)
print("mel_conf:", mel_conf)

transformer = DiT(
    **arch,
    text_num_embeds=text_num_embeds,
    mel_dim=mel_dim,
)

nnet = CFM(
    transformer=transformer,
    audio_drop_prob=model_conf.get("audio_drop_prob", 0.0),
    cond_drop_prob=model_conf.get("cond_drop_prob", 0.0),
    num_channels=mel_dim,
    mel_spec_kwargs=mel_conf,
)

missing, unexpected = nnet.load_state_dict(state, strict=False)

print("missing:", missing[:20], "count:", len(missing))
print("unexpected:", unexpected[:20], "count:", len(unexpected))

if len(missing) == 0 and len(unexpected) == 0:
    print("pretrained load: STRICT-COMPATIBLE")
else:
    print("pretrained load: NON-STRICT loaded, check missing/unexpected above")

nnet = nnet.to(device)
nnet.eval()

# 작은 mel-domain forward test
B, N, D = 1, 80, mel_dim
noisy = torch.randn(B, N, D, device=device)
clean = torch.randn(B, N, D, device=device)
text = [" "]

with torch.no_grad():
    loss, cond, pred = nnet(inp=noisy, clean=clean, text=text)

print("loss:", float(loss))
print("cond:", cond.shape)
print("pred:", pred.shape)
