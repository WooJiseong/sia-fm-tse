"""
Waveform export/evaluation script for PN-conditioned FlowSE.

Note:
FlowSE+Vocos generated waveforms may not be sample-level aligned with the
target waveform. Therefore SI-SNR/SI-SDR values from this script should be
treated as diagnostic, not as final paper metrics, unless alignment is handled.
"""

import glob
import json
import os
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from vocos import Vocos
from vocos.feature_extractors import EncodecFeatures

from model import CFM, DiT
from model.pn_conditioner import PNConditionedCFM, PNDiT


EPS = np.finfo(np.float32).eps


class PNFullCacheDataset(Dataset):
    def __init__(self, cache_dir, max_samples=None):
        self.files = sorted(glob.glob(os.path.join(cache_dir, "sample_*.pt")))
        if max_samples is not None:
            self.files = self.files[: int(max_samples)]
        if len(self.files) == 0:
            raise RuntimeError(f"No cache files found in {cache_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = torch.load(self.files[idx], map_location="cpu")

        mix_wave = item["mix_wave"].squeeze().float()
        target_wave = item["target_wave"].squeeze().float()
        cond_emb = item["cond_emb"].squeeze(0).float()
        sample_rate = int(item["sample_rate"])

        return {
            "utt_id": Path(self.files[idx]).stem,
            "mix_wave": mix_wave,
            "target_wave": target_wave,
            "cond_emb": cond_emb,
            "sample_rate": sample_rate,
        }


def collate_fn(batch):
    mix_lens = {item["mix_wave"].numel() for item in batch}
    tgt_lens = {item["target_wave"].numel() for item in batch}
    if len(mix_lens) != 1 or len(tgt_lens) != 1:
        raise ValueError("Variable-length cache samples are not supported by this evaluator")

    return {
        "utt_id": [item["utt_id"] for item in batch],
        "mix_wave": torch.stack([item["mix_wave"] for item in batch]),
        "target_wave": torch.stack([item["target_wave"] for item in batch]),
        "cond_emb": torch.stack([item["cond_emb"] for item in batch]),
        "sample_rate": torch.tensor([item["sample_rate"] for item in batch], dtype=torch.long),
    }


def get_env(name, default, cast=str):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    if cast is bool:
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return cast(value)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def normalize(audio, target_level=-25):
    rms = (audio ** 2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    return scalar * audio


def align(ref, est):
    n = min(len(ref), len(est))
    return ref[:n], est[:n]


def snr(ref, est):
    ref, est = align(ref, est)
    noise = ref - est
    return 10.0 * np.log10((np.sum(ref ** 2) + EPS) / (np.sum(noise ** 2) + EPS))


def si_snr(ref, est):
    ref, est = align(ref, est)

    ref = ref - np.mean(ref)
    est = est - np.mean(est)

    target = np.sum(est * ref) * ref / (np.sum(ref ** 2) + EPS)
    noise = est - target

    return 10.0 * np.log10((np.sum(target ** 2) + EPS) / (np.sum(noise ** 2) + EPS))


def build_cfm(
    flowse_state,
    conf,
    device,
    use_pn=False,
    token_mode="freq_pool",
    max_full_tokens=None,
    injection_mode="output",
):
    model_conf = conf["model"]
    arch = dict(model_conf["arch"])
    mel_conf = dict(model_conf["mel_spec"])

    text_num_embeds = flowse_state["transformer.text_embed.text_embed.weight"].shape[0] - 1
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

    cfm.load_state_dict(flowse_state, strict=True)

    if use_pn:
        cfm.transformer = PNDiT(
            cfm.transformer,
            spk_dim=64,
            freq_bins=65,
            temporal_pool="mean_std",
            dropout=0.0,
            token_mode=token_mode,
            max_full_tokens=max_full_tokens,
            injection_mode=injection_mode,
        )

    return cfm.to(device).eval(), mel_conf


def decode_mel_to_16k(vocoder, mel, target_sr, out_sr):
    vocos_mel = mel.transpose(-1, -2).to(torch.float32)
    wav = vocoder.decode(vocos_mel).squeeze().detach().cpu().numpy()
    wav = normalize(wav)

    if target_sr != out_sr:
        wav = librosa.resample(wav, orig_sr=target_sr, target_sr=out_sr)

    return wav.astype(np.float32)


def main():
    seed = get_env("SEED", 12345, int)
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    cache_dir = get_env("CACHE_DIR", "../bridge_outputs/pn_full_cache_train_5000")
    flowse_ckpt = get_env("FLOWSE_CKPT", "wenetspeech4tts_Premium.pt.tar")
    adapter_ckpt = get_env("ADAPTER_CKPT", "output/pn_online_adapter_bs2_epoch10.pt")
    vocoder_dir = get_env("VOCODER_DIR", "vocos-mel-24khz")
    save_dir = get_env("SAVE_DIR", "waveform_eval_outputs")
    max_samples = get_env("MAX_SAMPLES", 10, int)
    batch_size = get_env("BATCH_SIZE", 1, int)
    steps = get_env("STEPS", 32, int)
    save_wavs = get_env("SAVE_WAVS", False, bool)
    cfg_strength = get_env("CFG_STRENGTH", 1.0, float)
    token_mode = get_env("TOKEN_MODE", "freq_pool")
    injection_mode = get_env("INJECTION_MODE", "output")
    max_full_tokens_env = os.environ.get("MAX_FULL_TOKENS", "")
    max_full_tokens = int(max_full_tokens_env) if max_full_tokens_env else None

    os.makedirs(save_dir, exist_ok=True)

    print("cache_dir:", cache_dir)
    print("flowse_ckpt:", flowse_ckpt)
    print("adapter_ckpt:", adapter_ckpt)
    print("vocoder_dir:", vocoder_dir)
    print("max_samples:", max_samples)
    print("batch_size:", batch_size)
    print("steps:", steps)
    print("save_wavs:", save_wavs)
    print("token_mode:", token_mode)
    print("injection_mode:", injection_mode)
    print("max_full_tokens:", max_full_tokens)

    dataset = PNFullCacheDataset(cache_dir, max_samples=max_samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_fn,
    )

    ckpt = torch.load(flowse_ckpt, map_location="cpu")
    flowse_state = ckpt["model_state_dict"]

    with open("config/train.yaml", "r") as f:
        conf = yaml.safe_load(f)

    baseline_cfm, mel_conf = build_cfm(flowse_state, conf, device, use_pn=False)
    pn_cfm, _ = build_cfm(
        flowse_state,
        conf,
        device,
        use_pn=True,
        token_mode=token_mode,
        max_full_tokens=max_full_tokens,
    )

    adapter = torch.load(adapter_ckpt, map_location="cpu")
    pn_cfm.transformer.pn_projector.load_state_dict(adapter["pn_projector"])
    pn_cfm.transformer.speaker_attn.load_state_dict(adapter["speaker_attn"])
    pn_cfm.transformer.query_proj.load_state_dict(adapter["query_proj"])
    pn_cfm.transformer.out_proj.load_state_dict(adapter["out_proj"])

    with torch.no_grad():
        pn_cfm.transformer.speaker_attn_gate.copy_(adapter["speaker_attn_gate"].to(device))

    vocoder = load_vocoder(vocoder_dir, device)
    target_sr = int(mel_conf["target_sample_rate"])

    print("loaded FlowSE epoch:", ckpt.get("epoch"))
    print("loaded adapter epoch:", adapter.get("epoch"))
    print("loaded adapter loss:", adapter.get("epoch_loss"))
    print("loaded adapter gate:", adapter.get("speaker_attn_gate"))
    print("target_sr:", target_sr)

    results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):
            mix_wave = batch["mix_wave"].to(device)
            target_wave = batch["target_wave"].to(device)
            cond_emb = batch["cond_emb"].to(device)
            sample_rates = batch["sample_rate"]
            utt_ids = batch["utt_id"]

            sr = int(sample_rates[0])
            if not torch.all(sample_rates == sr):
                raise ValueError("Mixed sample rates in one batch are not supported")

            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr).to(device)
                mix_for_flowse = resampler(mix_wave)
            else:
                mix_for_flowse = mix_wave

            text = [" "] * mix_for_flowse.shape[0]

            seed_base = seed + batch_idx

            torch.manual_seed(seed_base)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_base)

            base_mel, _ = baseline_cfm.sample(
                cond=mix_for_flowse,
                text=text,
                drop_text=True,
                steps=steps,
                cfg_strength=cfg_strength,
            )

            torch.manual_seed(seed_base)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_base)

            pn_cfm.transformer.set_condition(cond_emb)
            try:
                pn_mel, _ = pn_cfm.sample(
                    cond=mix_for_flowse,
                    text=text,
                    drop_text=True,
                    steps=steps,
                    cfg_strength=cfg_strength,
                )
            finally:
                pn_cfm.transformer.clear_condition()

            for i, utt_id in enumerate(utt_ids):
                tgt_np = target_wave[i].detach().cpu().numpy()
                mix_np = mix_wave[i].detach().cpu().numpy()

                base_np = decode_mel_to_16k(vocoder, base_mel[i : i + 1], target_sr, sr)
                pn_np = decode_mel_to_16k(vocoder, pn_mel[i : i + 1], target_sr, sr)

                row = {
                    "utt_id": utt_id,
                    "mix_snr": snr(tgt_np, mix_np),
                    "base_snr": snr(tgt_np, base_np),
                    "pn_snr": snr(tgt_np, pn_np),
                    "mix_si_snr": si_snr(tgt_np, mix_np),
                    "base_si_snr": si_snr(tgt_np, base_np),
                    "pn_si_snr": si_snr(tgt_np, pn_np),
                }

                row["base_imp_snr"] = row["base_snr"] - row["mix_snr"]
                row["pn_imp_snr"] = row["pn_snr"] - row["mix_snr"]
                row["base_imp_si_snr"] = row["base_si_snr"] - row["mix_si_snr"]
                row["pn_imp_si_snr"] = row["pn_si_snr"] - row["mix_si_snr"]

                results.append(row)

                if save_wavs:
                    wav_dir = os.path.join(save_dir, "wavs", utt_id)
                    os.makedirs(wav_dir, exist_ok=True)
                    sf.write(os.path.join(wav_dir, "mixture.wav"), mix_np, sr, subtype="PCM_16")
                    sf.write(os.path.join(wav_dir, "target_clean.wav"), tgt_np, sr, subtype="PCM_16")
                    sf.write(os.path.join(wav_dir, "flowse_baseline.wav"), base_np, sr, subtype="PCM_16")
                    sf.write(os.path.join(wav_dir, "pn_flowse.wav"), pn_np, sr, subtype="PCM_16")

    def summarize(key):
        arr = np.array([r[key] for r in results], dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    summary = {
        "samples": len(results),
        "steps": steps,
        "cfg_strength": cfg_strength,
    }

    for key in [
        "mix_snr",
        "base_snr",
        "pn_snr",
        "base_imp_snr",
        "pn_imp_snr",
        "mix_si_snr",
        "base_si_snr",
        "pn_si_snr",
        "base_imp_si_snr",
        "pn_imp_si_snr",
    ]:
        mean, std = summarize(key)
        summary[key] = {"mean": mean, "std": std}

    print("===== Waveform Diagnostic Evaluation =====")
    print(json.dumps(summary, indent=2))

    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(save_dir, "per_sample.jsonl"), "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    print("saved:", save_dir)


if __name__ == "__main__":
    main()
