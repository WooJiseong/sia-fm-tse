import os


# PN tokenization mode for adapter evaluation
TOKEN_MODE = os.environ.get("TOKEN_MODE", "freq_pool")
INJECTION_MODE = os.environ.get("INJECTION_MODE", "output")
MAX_FULL_TOKENS_ENV = os.environ.get("MAX_FULL_TOKENS", "")
MAX_FULL_TOKENS = int(MAX_FULL_TOKENS_ENV) if MAX_FULL_TOKENS_ENV else None
import glob
import torch
import torchaudio
import yaml
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from model import DiT, CFM
from model.pn_conditioner import PNConditionedCFM, PNDiT


class PNFullCacheDataset(Dataset):
    def __init__(self, cache_dir, max_samples=50):
        self.files = sorted(glob.glob(os.path.join(cache_dir, "sample_*.pt")))
        if max_samples is not None:
            self.files = self.files[:max_samples]
        if len(self.files) == 0:
            raise RuntimeError(f"No cache files found in {cache_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = torch.load(self.files[idx], map_location="cpu")

        mix_wave = item["mix_wave"].squeeze(0).squeeze(0)
        target_wave = item["target_wave"].squeeze(0).squeeze(0)
        cond_emb = item["cond_emb"].squeeze(0)
        sr = item["sample_rate"]

        return mix_wave, target_wave, cond_emb, sr


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "50"))
ADAPTER_CKPT = os.environ.get("ADAPTER_CKPT", "output/pn_adapter_bs2_epoch3.pt")
CACHE_DIR = os.environ.get("CACHE_DIR", "../bridge_outputs/pn_full_cache_train_5000")

# ===== Load pretrained FlowSE =====
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

cfm.transformer = PNDiT(
    cfm.transformer,
    spk_dim=64,
    freq_bins=65,
    temporal_pool="mean_std",
    token_mode=TOKEN_MODE,
    max_full_tokens=MAX_FULL_TOKENS,
)

cfm = cfm.to(device)
cfm.eval()

for p in cfm.parameters():
    p.requires_grad = False

model = PNConditionedCFM(cfm=cfm).to(device)

# ===== Load trained PN adapter =====
adapter = torch.load(ADAPTER_CKPT, map_location="cpu")
model.cfm.transformer.pn_projector.load_state_dict(adapter["pn_projector"])
model.cfm.transformer.speaker_attn.load_state_dict(adapter["speaker_attn"])

if "query_proj" in adapter:
    model.cfm.transformer.query_proj.load_state_dict(adapter["query_proj"])
if "out_proj" in adapter:
    model.cfm.transformer.out_proj.load_state_dict(adapter["out_proj"])
if "speaker_attn_gate" in adapter:
    model.cfm.transformer.speaker_attn_gate.data.copy_(adapter["speaker_attn_gate"])

model.eval()

print("loaded FlowSE checkpoint:", ckpt_path)
print("loaded adapter:", ADAPTER_CKPT)
print("TOKEN_MODE:", TOKEN_MODE)
print("INJECTION_MODE:", INJECTION_MODE)
print("MAX_FULL_TOKENS:", MAX_FULL_TOKENS)
print("adapter epoch:", adapter.get("epoch"))
print("adapter epoch_loss:", adapter.get("epoch_loss"))
print("cache_dir:", CACHE_DIR)

dataset = PNFullCacheDataset(CACHE_DIR, max_samples=MAX_SAMPLES)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    drop_last=False,
)

baseline_losses = []
adapter_losses = []
mse_noisy_to_clean = []

with torch.no_grad():
    for batch_idx, (mix_wave, target_wave, cond_emb, sr) in enumerate(tqdm(loader)):
        mix_wave = mix_wave.to(device)
        target_wave = target_wave.to(device)
        cond_emb = cond_emb.to(device)

        if int(sr[0]) != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=int(sr[0]),
                new_freq=target_sr,
            ).to(device)
            mix_wave = resampler(mix_wave)
            target_wave = resampler(target_wave)

        noisy_mel = cfm.mel_spec(mix_wave).permute(0, 2, 1)
        clean_mel = cfm.mel_spec(target_wave).permute(0, 2, 1)
        text = [" "] * noisy_mel.shape[0]

        # baseline: PN condition 없이 pretrained FlowSE만 사용
        seed = 12345 + batch_idx

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        base_loss, base_cond, base_pred = model.cfm(
            inp=noisy_mel,
            clean=clean_mel,
            text=text,
        )

        # adapter: PN condition 사용
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        pn_loss, pn_cond, pn_pred = model(
            inp=noisy_mel,
            clean=clean_mel,
            text=text,
            cond_emb=cond_emb,
        )

        mse_noisy = torch.mean((noisy_mel - clean_mel) ** 2, dim=(1, 2))

        baseline_losses.append(base_loss.item())
        adapter_losses.append(pn_loss.item())
        mse_noisy_to_clean.extend(mse_noisy.detach().cpu().tolist())

baseline_losses = np.array(baseline_losses)
adapter_losses = np.array(adapter_losses)
mse_noisy_to_clean = np.array(mse_noisy_to_clean)

print("===== Full PN Adapter Mel-domain Evaluation =====")
print("samples:", len(dataset))
print("MSE(noisy_mel, clean_mel):", mse_noisy_to_clean.mean(), mse_noisy_to_clean.std())
print("FlowSE baseline loss:", baseline_losses.mean(), baseline_losses.std())
print("PN adapter loss:", adapter_losses.mean(), adapter_losses.std())
print("loss improvement:", (baseline_losses - adapter_losses).mean())
