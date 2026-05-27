import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

FLOWSE_ROOT = os.path.abspath(os.environ.get("FLOWSE_ROOT", os.getcwd()))
PN_REPO = os.environ.get(
    "PN_REPO",
    "/gpfs/home1/eri0529/SIA/paper/TSE-through-Positive-Negative-Enroll",
)


def import_flowse_modules(flowse_root: str, pn_repo: str):
    """Import FlowSE modules robustly across slightly different repo layouts.

    This handles three cases:
    1) official-style `from model import DiT, CFM`
    2) submodule-style `model.cfm.CFM`, `model.backbones.dit.DiT`
    3) local namespace-package clones where classes are located somewhere under
       `model/**/*.py` but are not exported from `model/__init__.py`.
    """
    import importlib
    import pkgutil
    import traceback

    flowse_root = os.path.abspath(flowse_root)
    pn_repo = os.path.abspath(pn_repo)
    model_dir = os.path.join(flowse_root, "model")

    if not os.path.isdir(model_dir):
        raise ImportError(f"FLOWSE_ROOT does not contain a model/ directory: {model_dir}")

    for k in list(sys.modules.keys()):
        if k == "model" or k.startswith("model."):
            del sys.modules[k]

    cleaned_path = []
    for p in sys.path:
        abs_p = os.path.abspath(p or os.getcwd())
        if abs_p == pn_repo:
            continue
        if abs_p not in cleaned_path:
            cleaned_path.append(abs_p)

    sys.path = [flowse_root] + [p for p in cleaned_path if p != flowse_root]

    import_errors = []

    def try_import(module_name):
        try:
            return importlib.import_module(module_name)
        except Exception as exc:
            import_errors.append((module_name, repr(exc)))
            return None

    flowse_model = try_import("model")
    if flowse_model is None:
        raise ImportError(
            f"Failed to import top-level FlowSE model package from {model_dir}. "
            f"First errors: {import_errors[:5]}"
        )

    def find_attr(attr_name, preferred_modules):
        # 1. Already exported from `model`
        if hasattr(flowse_model, attr_name):
            return getattr(flowse_model, attr_name)

        # 2. Common module locations
        for module_name in preferred_modules:
            mod = try_import(module_name)
            if mod is not None and hasattr(mod, attr_name):
                return getattr(mod, attr_name)

        # 3. Last resort: scan all importable modules under model/.
        #    This is intentionally broad because your local FlowSE clone is a
        #    namespace package and does not have model.cfm.
        for module_info in pkgutil.walk_packages([model_dir], prefix="model."):
            module_name = module_info.name
            mod = try_import(module_name)
            if mod is not None and hasattr(mod, attr_name):
                print(f"found FlowSE {attr_name} in {module_name}", flush=True)
                return getattr(mod, attr_name)

        raise ImportError(
            f"Could not find `{attr_name}` anywhere under {model_dir}. "
            f"Run: find {model_dir} -maxdepth 3 -type f -name '*.py' -print"
        )

    DiT = find_attr(
        "DiT",
        [
            "model.backbones.dit",
            "model.dit",
            "model.transformer",
            "model.modules",
            "model.model",
        ],
    )
    CFM = find_attr(
        "CFM",
        [
            "model.cfm",
            "model.flow_matching",
            "model.modules",
            "model.model",
        ],
    )

    # Make namespace-package clones behave like official FlowSE's model package.
    setattr(flowse_model, "DiT", DiT)
    setattr(flowse_model, "CFM", CFM)

    try:
        pn_conditioner = importlib.import_module("model.pn_conditioner")
        model_utils = importlib.import_module("model.model_utils")
    except Exception as exc:
        raise ImportError(
            "Found FlowSE DiT/CFM, but failed to import model.pn_conditioner "
            "or model.model_utils. Make sure pn_conditioner.py is inside "
            f"{model_dir}. Original error: {repr(exc)}"
        ) from exc

    print("FlowSE import resolved:", flush=True)
    print("  FLOWSE_ROOT:", flowse_root, flush=True)
    print("  DiT:", DiT, flush=True)
    print("  CFM:", CFM, flush=True)

    return (
        DiT,
        CFM,
        pn_conditioner.PNConditionedCFM,
        pn_conditioner.PNDiT,
        model_utils.exists,
        model_utils.list_str_to_idx,
        model_utils.list_str_to_tensor,
    )



def get_env(name, default, cast=str):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return cast(value)


def tokenize_text(cfm, text, batch, device):
    if isinstance(text, list):
        if exists(cfm.vocab_char_map):
            text = list_str_to_idx(text, cfm.vocab_char_map).to(device)
        else:
            text = list_str_to_tensor(text).to(device)
        assert text.shape[0] == batch
    return text


def flow_step_metrics(cfm, noisy_mel, clean_mel, text, cond_emb=None):
    """Evaluate the same random flow point with and without PN condition."""
    batch = noisy_mel.shape[0]
    dtype = noisy_mel.dtype
    device = noisy_mel.device
    text = tokenize_text(cfm, text, batch, device)

    x1 = clean_mel
    x0 = torch.randn_like(x1)
    time = torch.rand((batch,), dtype=dtype, device=device)
    t = time.unsqueeze(-1).unsqueeze(-1)
    xt = (1 - t) * x0 + t * x1
    flow = x1 - x0

    kwargs = dict(
        x=xt,
        cond=noisy_mel,
        text=text,
        time=time,
        drop_audio_cond=False,
        drop_text=False,
    )
    if cond_emb is None:
        pred = cfm.transformer(**kwargs)
    else:
        # Use the same PN condition injection path as the training script.
        cfm.transformer.set_condition(cond_emb)
        try:
            pred = cfm.transformer(**kwargs)
        finally:
            cfm.transformer.clear_condition()

    cfm_loss = F.mse_loss(pred, flow, reduction="mean")
    clean_est = xt + (1 - t) * pred
    clean_l1 = F.l1_loss(clean_est, x1)
    return cfm_loss, clean_l1


def build_model(flowse_ckpt, adapter, device):
    ckpt = torch.load(flowse_ckpt, map_location="cpu")
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

    token_mode = os.environ.get("TOKEN_MODE", adapter.get("token_mode", "full_2d"))
    injection_mode = os.environ.get("INJECTION_MODE", adapter.get("injection_mode", "block"))
    max_full_tokens_env = os.environ.get("MAX_FULL_TOKENS", "")
    max_full_tokens = int(max_full_tokens_env) if max_full_tokens_env else adapter.get("max_full_tokens")

    cfm.transformer = PNDiT(
        cfm.transformer,
        spk_dim=64,
        freq_bins=65,
        temporal_pool=adapter.get("temporal_pool", "mean_std"),
        token_mode=token_mode,
        max_full_tokens=max_full_tokens,
        injection_mode=injection_mode,
    )

    cfm.transformer.pn_projector.load_state_dict(adapter["pn_projector"])
    cfm.transformer.speaker_attn.load_state_dict(adapter["speaker_attn"])
    cfm.transformer.query_proj.load_state_dict(adapter["query_proj"])
    cfm.transformer.out_proj.load_state_dict(adapter["out_proj"])
    if "speaker_attn_gate" in adapter:
        cfm.transformer.speaker_attn_gate.data.copy_(adapter["speaker_attn_gate"])

    flowse_tail = adapter.get("flowse_tail", {})
    if flowse_tail:
        base_state = cfm.transformer.base_dit.state_dict()
        base_state.update(flowse_tail)
        cfm.transformer.base_dit.load_state_dict(base_state, strict=True)

    return cfm.to(device).eval(), target_sr, {
        "token_mode": token_mode,
        "injection_mode": injection_mode,
        "max_full_tokens": max_full_tokens,
        "flowse_tail_params": len(flowse_tail),
        "flowse_epoch": ckpt.get("epoch"),
        "flowse_best_loss": ckpt.get("best_loss"),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    batch_size = get_env("BATCH_SIZE", 2, int)
    max_samples = get_env("MAX_SAMPLES", 50, int)
    flowse_ckpt = get_env("FLOWSE_CKPT", "wenetspeech4tts_Premium.pt.tar")
    adapter_ckpt = get_env("ADAPTER_CKPT", "output/best.pt")
    cache_dir = get_env("CACHE_DIR", "../bridge_outputs/pn_full_cache_train_5000")

    adapter = torch.load(adapter_ckpt, map_location="cpu")
    cfm, target_sr, meta = build_model(flowse_ckpt, adapter, device)

    print("loaded FlowSE checkpoint:", flowse_ckpt)
    print("loaded adapter:", adapter_ckpt)
    print("TOKEN_MODE:", meta["token_mode"])
    print("INJECTION_MODE:", meta["injection_mode"])
    print("MAX_FULL_TOKENS:", meta["max_full_tokens"])
    print("loaded flowse_tail params:", meta["flowse_tail_params"])
    print("adapter epoch:", adapter.get("epoch"))
    print("adapter epoch_loss:", adapter.get("epoch_loss"))
    print("adapter epoch_cfm_loss:", adapter.get("epoch_cfm_loss"))
    print("adapter epoch_clean_est_l1:", adapter.get("epoch_clean_est_l1"))
    print("adapter clean_est_l1_weight:", adapter.get("clean_est_l1_weight"))
    print("cache_dir:", cache_dir)

    dataset = PNFullCacheDataset(cache_dir, max_samples=max_samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    baseline_losses = []
    adapter_losses = []
    baseline_clean_l1 = []
    adapter_clean_l1 = []
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

            seed = 12345 + batch_idx
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            base_loss, base_l1 = flow_step_metrics(
                cfm,
                noisy_mel=noisy_mel,
                clean_mel=clean_mel,
                text=text,
                cond_emb=None,
            )

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            pn_loss, pn_l1 = flow_step_metrics(
                cfm,
                noisy_mel=noisy_mel,
                clean_mel=clean_mel,
                text=text,
                cond_emb=cond_emb,
            )

            mse_noisy = torch.mean((noisy_mel - clean_mel) ** 2, dim=(1, 2))
            baseline_losses.append(base_loss.item())
            adapter_losses.append(pn_loss.item())
            baseline_clean_l1.append(base_l1.item())
            adapter_clean_l1.append(pn_l1.item())
            mse_noisy_to_clean.extend(mse_noisy.detach().cpu().tolist())

    baseline_losses = np.array(baseline_losses)
    adapter_losses = np.array(adapter_losses)
    baseline_clean_l1 = np.array(baseline_clean_l1)
    adapter_clean_l1 = np.array(adapter_clean_l1)
    mse_noisy_to_clean = np.array(mse_noisy_to_clean)

    print("===== PN Adapter Mel-domain Diagnostic Evaluation =====")
    print("samples:", len(dataset))
    print("MSE(noisy_mel, clean_mel):", mse_noisy_to_clean.mean(), mse_noisy_to_clean.std())
    print("FlowSE baseline CFM loss:", baseline_losses.mean(), baseline_losses.std())
    print("PN adapter CFM loss:", adapter_losses.mean(), adapter_losses.std())
    print("CFM loss improvement:", (baseline_losses - adapter_losses).mean())
    print("FlowSE baseline clean-est L1:", baseline_clean_l1.mean(), baseline_clean_l1.std())
    print("PN adapter clean-est L1:", adapter_clean_l1.mean(), adapter_clean_l1.std())
    print("clean-est L1 improvement:", (baseline_clean_l1 - adapter_clean_l1).mean())


if __name__ == "__main__":
    main()
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

FLOWSE_ROOT = os.path.abspath(os.environ.get("FLOWSE_ROOT", os.getcwd()))
PN_REPO = os.environ.get(
    "PN_REPO",
    "/gpfs/home1/eri0529/SIA/paper/TSE-through-Positive-Negative-Enroll",
)


def import_flowse_modules(flowse_root: str, pn_repo: str):
    """Import FlowSE modules even when `model` is a namespace package.

    Some local clones do not expose DiT/CFM from model/__init__.py, so
    `from model import DiT, CFM` can fail even though these classes exist in
    model.backbones.dit and model.cfm. We import the concrete submodules and
    then attach DiT/CFM to the loaded `model` package so local helper modules
    such as model.pn_conditioner can also resolve them.
    """
    import importlib

    flowse_root = os.path.abspath(flowse_root)
    pn_repo = os.path.abspath(pn_repo)

    # Remove previously imported FlowSE/PN `model` modules. Both repos use the
    # top-level package name `model`, so stale modules can silently point to the
    # wrong repository.
    for k in list(sys.modules.keys()):
        if k == "model" or k.startswith("model."):
            del sys.modules[k]

    cleaned_path = []
    for p in sys.path:
        abs_p = os.path.abspath(p or os.getcwd())
        if abs_p == pn_repo:
            continue
        if abs_p not in cleaned_path:
            cleaned_path.append(abs_p)

    sys.path = [flowse_root] + [p for p in cleaned_path if p != flowse_root]

    try:
        flowse_model = importlib.import_module("model")
        cfm_mod = importlib.import_module("model.cfm")
        dit_mod = importlib.import_module("model.backbones.dit")

        CFM = cfm_mod.CFM
        DiT = dit_mod.DiT

        # Make namespace-package clones behave like the official package
        # __init__.py, which exports CFM and DiT.
        setattr(flowse_model, "CFM", CFM)
        setattr(flowse_model, "DiT", DiT)

        pn_conditioner = importlib.import_module("model.pn_conditioner")
        model_utils = importlib.import_module("model.model_utils")

    except Exception as exc:
        raise ImportError(
            "Failed to import FlowSE modules. Check that FLOWSE_ROOT points to "
            "the FlowSE repo root containing model/cfm.py, "
            "model/backbones/dit.py, and model/pn_conditioner.py. "
            f"FLOWSE_ROOT={flowse_root}, sys.path[0]={sys.path[0]}"
        ) from exc

    return (
        CFM,
        DiT,
        model_utils.exists,
        model_utils.list_str_to_idx,
        model_utils.list_str_to_tensor,
        pn_conditioner.PNDiT,
    )



CFM, DiT, exists, list_str_to_idx, list_str_to_tensor, PNDiT = import_flowse_modules(
    FLOWSE_ROOT, PN_REPO
)


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


def get_env(name, default, cast=str):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return cast(value)


def tokenize_text(cfm, text, batch, device):
    if isinstance(text, list):
        if exists(cfm.vocab_char_map):
            text = list_str_to_idx(text, cfm.vocab_char_map).to(device)
        else:
            text = list_str_to_tensor(text).to(device)
        assert text.shape[0] == batch
    return text


def flow_step_metrics(cfm, noisy_mel, clean_mel, text, cond_emb=None):
    """Evaluate the same random flow point with and without PN condition."""
    batch = noisy_mel.shape[0]
    dtype = noisy_mel.dtype
    device = noisy_mel.device
    text = tokenize_text(cfm, text, batch, device)

    x1 = clean_mel
    x0 = torch.randn_like(x1)
    time = torch.rand((batch,), dtype=dtype, device=device)
    t = time.unsqueeze(-1).unsqueeze(-1)
    xt = (1 - t) * x0 + t * x1
    flow = x1 - x0

    kwargs = dict(
        x=xt,
        cond=noisy_mel,
        text=text,
        time=time,
        drop_audio_cond=False,
        drop_text=False,
    )
    if cond_emb is None:
        pred = cfm.transformer(**kwargs)
    else:
        # Use the same PN condition injection path as the training script.
        cfm.transformer.set_condition(cond_emb)
        try:
            pred = cfm.transformer(**kwargs)
        finally:
            cfm.transformer.clear_condition()

    cfm_loss = F.mse_loss(pred, flow, reduction="mean")
    clean_est = xt + (1 - t) * pred
    clean_l1 = F.l1_loss(clean_est, x1)
    return cfm_loss, clean_l1


def build_model(flowse_ckpt, adapter, device):
    ckpt = torch.load(flowse_ckpt, map_location="cpu")
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

    token_mode = os.environ.get("TOKEN_MODE", adapter.get("token_mode", "full_2d"))
    injection_mode = os.environ.get("INJECTION_MODE", adapter.get("injection_mode", "block"))
    max_full_tokens_env = os.environ.get("MAX_FULL_TOKENS", "")
    max_full_tokens = int(max_full_tokens_env) if max_full_tokens_env else adapter.get("max_full_tokens")

    cfm.transformer = PNDiT(
        cfm.transformer,
        spk_dim=64,
        freq_bins=65,
        temporal_pool=adapter.get("temporal_pool", "mean_std"),
        token_mode=token_mode,
        max_full_tokens=max_full_tokens,
        injection_mode=injection_mode,
    )

    cfm.transformer.pn_projector.load_state_dict(adapter["pn_projector"])
    cfm.transformer.speaker_attn.load_state_dict(adapter["speaker_attn"])
    cfm.transformer.query_proj.load_state_dict(adapter["query_proj"])
    cfm.transformer.out_proj.load_state_dict(adapter["out_proj"])
    if "speaker_attn_gate" in adapter:
        cfm.transformer.speaker_attn_gate.data.copy_(adapter["speaker_attn_gate"])

    flowse_tail = adapter.get("flowse_tail", {})
    if flowse_tail:
        base_state = cfm.transformer.base_dit.state_dict()
        base_state.update(flowse_tail)
        cfm.transformer.base_dit.load_state_dict(base_state, strict=True)

    return cfm.to(device).eval(), target_sr, {
        "token_mode": token_mode,
        "injection_mode": injection_mode,
        "max_full_tokens": max_full_tokens,
        "flowse_tail_params": len(flowse_tail),
        "flowse_epoch": ckpt.get("epoch"),
        "flowse_best_loss": ckpt.get("best_loss"),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    batch_size = get_env("BATCH_SIZE", 2, int)
    max_samples = get_env("MAX_SAMPLES", 50, int)
    flowse_ckpt = get_env("FLOWSE_CKPT", "wenetspeech4tts_Premium.pt.tar")
    adapter_ckpt = get_env("ADAPTER_CKPT", "output/best.pt")
    cache_dir = get_env("CACHE_DIR", "../bridge_outputs/pn_full_cache_train_5000")

    adapter = torch.load(adapter_ckpt, map_location="cpu")
    cfm, target_sr, meta = build_model(flowse_ckpt, adapter, device)

    print("loaded FlowSE checkpoint:", flowse_ckpt)
    print("loaded adapter:", adapter_ckpt)
    print("TOKEN_MODE:", meta["token_mode"])
    print("INJECTION_MODE:", meta["injection_mode"])
    print("MAX_FULL_TOKENS:", meta["max_full_tokens"])
    print("loaded flowse_tail params:", meta["flowse_tail_params"])
    print("adapter epoch:", adapter.get("epoch"))
    print("adapter epoch_loss:", adapter.get("epoch_loss"))
    print("adapter epoch_cfm_loss:", adapter.get("epoch_cfm_loss"))
    print("adapter epoch_clean_est_l1:", adapter.get("epoch_clean_est_l1"))
    print("adapter clean_est_l1_weight:", adapter.get("clean_est_l1_weight"))
    print("cache_dir:", cache_dir)

    dataset = PNFullCacheDataset(cache_dir, max_samples=max_samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    baseline_losses = []
    adapter_losses = []
    baseline_clean_l1 = []
    adapter_clean_l1 = []
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

            seed = 12345 + batch_idx
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            base_loss, base_l1 = flow_step_metrics(
                cfm,
                noisy_mel=noisy_mel,
                clean_mel=clean_mel,
                text=text,
                cond_emb=None,
            )

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            pn_loss, pn_l1 = flow_step_metrics(
                cfm,
                noisy_mel=noisy_mel,
                clean_mel=clean_mel,
                text=text,
                cond_emb=cond_emb,
            )

            mse_noisy = torch.mean((noisy_mel - clean_mel) ** 2, dim=(1, 2))
            baseline_losses.append(base_loss.item())
            adapter_losses.append(pn_loss.item())
            baseline_clean_l1.append(base_l1.item())
            adapter_clean_l1.append(pn_l1.item())
            mse_noisy_to_clean.extend(mse_noisy.detach().cpu().tolist())

    baseline_losses = np.array(baseline_losses)
    adapter_losses = np.array(adapter_losses)
    baseline_clean_l1 = np.array(baseline_clean_l1)
    adapter_clean_l1 = np.array(adapter_clean_l1)
    mse_noisy_to_clean = np.array(mse_noisy_to_clean)

    print("===== PN Adapter Mel-domain Diagnostic Evaluation =====")
    print("samples:", len(dataset))
    print("MSE(noisy_mel, clean_mel):", mse_noisy_to_clean.mean(), mse_noisy_to_clean.std())
    print("FlowSE baseline CFM loss:", baseline_losses.mean(), baseline_losses.std())
    print("PN adapter CFM loss:", adapter_losses.mean(), adapter_losses.std())
    print("CFM loss improvement:", (baseline_losses - adapter_losses).mean())
    print("FlowSE baseline clean-est L1:", baseline_clean_l1.mean(), baseline_clean_l1.std())
    print("PN adapter clean-est L1:", adapter_clean_l1.mean(), adapter_clean_l1.std())
    print("clean-est L1 improvement:", (baseline_clean_l1 - adapter_clean_l1).mean())


if __name__ == "__main__":
    main()