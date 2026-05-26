import os
import torch
import torchaudio
import yaml
import soundfile as sf
import librosa
import numpy as np

from vocos import Vocos
from vocos.feature_extractors import EncodecFeatures

from model import DiT, CFM
from model.pn_conditioner import PNConditionedCFM, PNDiT

EPS = np.finfo(float).eps


def normalize(audio, target_level=-25):
    rms = (audio ** 2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    return scalar * audio


def load_vocoder(local_path, device):
    config_path = os.path.join(local_path, "config.yaml")
    model_path = os.path.join(local_path, "pytorch_model.bin")

    vocoder = Vocos.from_hparams(config_path)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    if isinstance(vocoder.feature_extractor, EncodecFeatures):
        encodec_parameters = {
            "feature_extractor.encodec." + key: value
            for key, value in vocoder.feature_extractor.encodec.state_dict().items()
        }
        state_dict.update(encodec_parameters)

    vocoder.load_state_dict(state_dict)
    return vocoder.eval().to(device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

FLOWSE_CKPT = "/gpfs/home1/eri0529/SIA/paper/flowse/wenetspeech4tts_Premium.pt.tar"
ADAPTER_CKPT = "output/pn_online_adapter_bs2_epoch10.pt"
CACHE_FILE = "/gpfs/home1/eri0529/SIA/paper/bridge_outputs/pn_full_cache_train_5000/sample_000000.pt"
VOCODER_DIR = "vocos-mel-24khz"
SAVE_DIR = "samples_pn_flowse"

os.makedirs(SAVE_DIR, exist_ok=True)

item = torch.load(CACHE_FILE, map_location="cpu")

mix_wave = item["mix_wave"].squeeze(0).squeeze(0).float().to(device)
target_wave = item["target_wave"].squeeze(0).squeeze(0).float().to(device)
cond_emb = item["cond_emb"].squeeze(0).float().unsqueeze(0).to(device)
sr = int(item["sample_rate"])

print("mix_wave:", mix_wave.shape, "sr:", sr)
print("target_wave:", target_wave.shape)
print("cond_emb:", cond_emb.shape)

sf.write(os.path.join(SAVE_DIR, "sample000_mixture.wav"), mix_wave.detach().cpu().numpy(), sr)
sf.write(os.path.join(SAVE_DIR, "sample000_target_clean.wav"), target_wave.detach().cpu().numpy(), sr)

ckpt = torch.load(FLOWSE_CKPT, map_location="cpu")
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
    dropout=0.0,
)

cfm = cfm.to(device).eval()

adapter = torch.load(ADAPTER_CKPT, map_location="cpu")

cfm.transformer.pn_projector.load_state_dict(adapter["pn_projector"])
cfm.transformer.speaker_attn.load_state_dict(adapter["speaker_attn"])
cfm.transformer.query_proj.load_state_dict(adapter["query_proj"])
cfm.transformer.out_proj.load_state_dict(adapter["out_proj"])

with torch.no_grad():
    cfm.transformer.speaker_attn_gate.copy_(adapter["speaker_attn_gate"].to(device))

model = PNConditionedCFM(cfm=cfm).to(device).eval()

print("loaded FlowSE:", FLOWSE_CKPT)
print("loaded adapter:", ADAPTER_CKPT)
print("adapter epoch:", adapter.get("epoch"))
print("adapter loss:", adapter.get("epoch_loss"))
print("adapter gate:", adapter.get("speaker_attn_gate"))

vocoder = load_vocoder(VOCODER_DIR, device)
print("loaded vocoder:", VOCODER_DIR)

if sr != target_sr:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr).to(device)
    mix_24k = resampler(mix_wave.unsqueeze(0))
else:
    mix_24k = mix_wave.unsqueeze(0)

print("mix_24k:", mix_24k.shape)

text = [" "]

with torch.no_grad():
    baseline_mel, _ = cfm.sample(cond=mix_24k, text=text, drop_text=True)
    pn_mel, _ = model.cfm.sample(cond=mix_24k, text=text, cond_emb=cond_emb, drop_text=True)

    baseline_vocos_mel = baseline_mel.transpose(-1, -2).to(torch.float32)
    pn_vocos_mel = pn_mel.transpose(-1, -2).to(torch.float32)

    baseline_wave = vocoder.decode(baseline_vocos_mel).squeeze().detach().cpu().numpy()
    pn_wave = vocoder.decode(pn_vocos_mel).squeeze().detach().cpu().numpy()

baseline_wave = normalize(baseline_wave)
pn_wave = normalize(pn_wave)

baseline_wave_16k = librosa.resample(baseline_wave, orig_sr=target_sr, target_sr=sr)
pn_wave_16k = librosa.resample(pn_wave, orig_sr=target_sr, target_sr=sr)

sf.write(os.path.join(SAVE_DIR, "sample000_flowse_baseline.wav"), baseline_wave_16k, sr)
sf.write(os.path.join(SAVE_DIR, "sample000_pn_flowse_online.wav"), pn_wave_16k, sr)

print("saved dir:", SAVE_DIR)
print("saved:")
print(" - sample000_mixture.wav")
print(" - sample000_target_clean.wav")
print(" - sample000_flowse_baseline.wav")
print(" - sample000_pn_flowse_online.wav")
