from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio
import yaml
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:

    class SummaryWriter:  # type: ignore[no-redef]
        """No-op TensorBoard writer for bare training environments."""

        def __init__(self, *args, **kwargs):
            print("tensorboard is not installed; scalar logging is disabled")

        def add_scalar(self, *args, **kwargs):
            return None

        def flush(self):
            return None

        def close(self):
            return None

from sia_fm_tse.flowse.model import CFM, DiT
from sia_fm_tse.flowse.model.model_utils import (
    exists,
    list_str_to_idx,
    list_str_to_tensor,
)
from sia_fm_tse.flowse.model.pn_conditioner import PNConditionedCFM, PNDiT
from sia_fm_tse.pnenroll.dataset import LibriDataset_single_emb
from sia_fm_tse.pnenroll.encoder import load_frozen_pn_encoder


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _require_path(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_dataset(conf: dict[str, Any]) -> LibriDataset_single_emb:
    paths = conf["paths"]
    data = conf["data"]
    sample_rate = int(data["sample_rate"])
    wave_length = int(float(data["wave_seconds"]) * sample_rate)

    # Fail early if the run accidentally points at _test_data or a missing split.
    librispeech_train = _require_path(paths["librispeech_train"], "librispeech_train")
    wham_noise_train = _require_path(paths["wham_noise_train"], "wham_noise_train")

    return LibriDataset_single_emb(
        str(librispeech_train),
        sample_rate=sample_rate,
        wave_length=wave_length,
        pos_example_length=wave_length,
        neg_example_length=wave_length,
        snr_db_range=list(data["snr_db_range"]),
        source_num=int(data["source_num"]),
        min_source_num=int(data["min_source_num"]),
        enroll_num=int(data["enroll_num"]),
        min_enroll_num=int(data["min_enroll_num"]),
        active_num=list(data["active_num"]),
        reproducable=False,
        normalize=False,
        filling_pattern=str(data["filling_pattern"]),
        return_dvec=False,
        dvec_rate=int(data["dvec_rate"]),
        include_silent=False,
        special_spk=list(data["special_spk"]),
        partial_range=list(data["partial_range"]),
        neg_partial_range=list(data["neg_partial_range"]),
        reverb=str(data["reverb"]),
        binaural=bool(data["binaural"]),
        reverb_cond=bool(data["reverb_cond"]),
        zero_in_tgt=bool(data["zero_in_tgt"]),
        noise_dir=str(wham_noise_train),
        same_disturb=bool(data["same_disturb"]),
    )


def _load_flowse_arch(conf: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    flowse = conf["flowse"]
    if flowse["init"] != "pretrained":
        return dict(flowse["arch"]), dict(flowse["mel_spec"])

    # Pretrained checkpoints must use the exact architecture they were saved with.
    flowse_config = _read_yaml(conf["paths"]["flowse_config"])
    model_conf = flowse_config["model"]
    return dict(model_conf["arch"]), dict(model_conf["mel_spec"])


def _build_flowse(conf: dict[str, Any], device: torch.device) -> CFM:
    flowse = conf["flowse"]
    paths = conf["paths"]
    init_mode = str(flowse["init"])
    arch, mel_spec = _load_flowse_arch(conf)
    mel_dim = int(mel_spec["n_mel_channels"])

    checkpoint = None
    if init_mode == "pretrained":
        # Infer text vocab size from checkpoint so local config mismatches do not break load.
        flowse_checkpoint = _require_path(paths["flowse_checkpoint"], "flowse_checkpoint")
        checkpoint = torch.load(flowse_checkpoint, map_location="cpu")
        state = checkpoint["model_state_dict"]
        text_num_embeds = (
            state["transformer.text_embed.text_embed.weight"].shape[0] - 1
        )
    elif init_mode == "scratch":
        state = None
        text_num_embeds = int(flowse["text_num_embeds"])
    else:
        raise ValueError("flowse.init must be one of: scratch, pretrained")

    transformer = DiT(
        **arch,
        text_num_embeds=text_num_embeds,
        mel_dim=mel_dim,
    )
    cfm = CFM(
        transformer=transformer,
        audio_drop_prob=float(flowse["audio_drop_prob"]),
        cond_drop_prob=float(flowse["cond_drop_prob"]),
        num_channels=mel_dim,
        mel_spec_kwargs=mel_spec,
    )

    if state is not None:
        cfm.load_state_dict(state, strict=bool(flowse["load_strict"]))
        print("loaded pretrained FlowSE")
        print("checkpoint epoch:", checkpoint.get("epoch"))
        print("checkpoint best_loss:", checkpoint.get("best_loss"))
    else:
        print("initialized compact FlowSE from scratch")

    cfm.transformer = PNDiT(
        cfm.transformer,
        spk_dim=64,
        freq_bins=65,
        temporal_pool="mean_std",
        dropout=float(flowse["pn_dropout"]),
        token_mode=str(flowse["token_mode"]),
        max_full_tokens=flowse["max_pn_tokens"],
        injection_mode=str(flowse["injection_mode"]),
    )
    return cfm.to(device)


def _set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def _unfreeze_flowse_base(pn_dit: PNDiT, mode: str, num_blocks: int) -> list[torch.nn.Parameter]:
    mode = mode.lower()
    if mode not in {"none", "tail", "all"}:
        raise ValueError("flowse.train_mode must be one of: none, tail, all")

    base = pn_dit.base_dit
    if mode == "none":
        return []

    if mode == "all":
        _set_requires_grad(base, True)
        base.train()
        return list(base.parameters())

    trainable: list[torch.nn.Parameter] = []
    if num_blocks <= 0:
        return trainable

    for block in base.transformer_blocks[-num_blocks:]:
        _set_requires_grad(block, True)
        block.train()
        trainable.extend(block.parameters())

    for module_name in ("input_embed", "norm_out", "proj_out"):
        module = getattr(base, module_name, None)
        if module is not None:
            _set_requires_grad(module, True)
            module.train()
            trainable.extend(module.parameters())

    return trainable


def _configure_trainable(model: PNConditionedCFM, conf: dict[str, Any]):
    flowse = conf["flowse"]
    train = conf["train"]
    cfm = model.cfm

    # Start frozen, then explicitly open the adapter and optional decoder subset.
    _set_requires_grad(cfm, False)

    adapter_modules = [
        cfm.transformer.pn_projector,
        cfm.transformer.speaker_attn,
        cfm.transformer.query_proj,
        cfm.transformer.out_proj,
    ]
    for module in adapter_modules:
        _set_requires_grad(module, True)
        module.train()
    cfm.transformer.speaker_attn_gate.requires_grad = True

    flowse_params = _unfreeze_flowse_base(
        cfm.transformer,
        str(flowse["train_mode"]),
        int(flowse["train_last_blocks"]),
    )
    adapter_params = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("cfm.transformer.base_dit.")
    ]

    param_groups = [
        {
            "params": adapter_params,
            "lr": float(train["lr"]),
            "weight_decay": float(train["weight_decay"]),
        }
    ]
    if flowse_params:
        param_groups.append(
            {
                "params": flowse_params,
                "lr": float(train["flowse_lr"]),
                "weight_decay": float(train["weight_decay"]),
            }
        )

    trainable_params = [p for group in param_groups for p in group["params"]]
    optimizer = torch.optim.AdamW(param_groups)
    return trainable_params, optimizer, adapter_params, flowse_params


def _tokenize_text(cfm: CFM, text: list[str], batch: int, device: torch.device):
    if exists(cfm.vocab_char_map):
        tokens = list_str_to_idx(text, cfm.vocab_char_map).to(device)
    else:
        tokens = list_str_to_tensor(text).to(device)
    assert tokens.shape[0] == batch
    return tokens


def _pn_flow_matching_step(
    cfm: CFM,
    noisy_mel: torch.Tensor,
    clean_mel: torch.Tensor,
    cond_emb: torch.Tensor,
):
    batch = noisy_mel.shape[0]
    dtype = noisy_mel.dtype
    device = cfm.device

    x1 = clean_mel
    x0 = torch.randn_like(x1)
    time = torch.rand((batch,), dtype=dtype, device=device)
    t = time.unsqueeze(-1).unsqueeze(-1)
    xt = (1 - t) * x0 + t * x1
    flow = x1 - x0

    drop_audio_cond = random.random() < cfm.audio_drop_prob
    drop_text = False
    if random.random() < cfm.cond_drop_prob:
        drop_audio_cond = True
        drop_text = True

    # We have no transcript for TSE, so the condition text is always blank.
    # When drop_text is True we still pass blank text and also set drop_text=True,
    # matching FlowSE's CFG convention explicitly.
    text = _tokenize_text(cfm, [" "] * batch, batch, device)

    cfm.transformer.set_condition(cond_emb)
    try:
        # PNDiT reads cond_emb from set_condition(), keeping CFM's call shape intact.
        pred = cfm.transformer(
            x=xt,
            cond=noisy_mel,
            text=text,
            time=time,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
        )
    finally:
        cfm.transformer.clear_condition()

    cfm_loss = F.mse_loss(pred, flow, reduction="mean")
    clean_est = xt + (1 - t) * pred
    clean_est_l1 = F.l1_loss(clean_est, x1)
    return cfm_loss, clean_est_l1, pred, clean_est


def _save_checkpoint(
    path: Path,
    model: PNConditionedCFM,
    optimizer: torch.optim.Optimizer,
    conf: dict[str, Any],
    epoch: int,
    global_step: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.cfm.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": conf,
            "metrics": metrics,
        },
        path,
    )


def train(config_path: str | Path, *, device_name: str | None = None, dry_run: bool = False) -> None:
    conf = _read_yaml(config_path)
    _seed_everything(int(conf["train"]["seed"]))

    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("device:", device)
    print("config:", config_path)
    print("librispeech_train:", conf["paths"]["librispeech_train"])
    print("wham_noise_train:", conf["paths"]["wham_noise_train"])

    dataset = _make_dataset(conf)
    loader = DataLoader(
        dataset,
        batch_size=int(conf["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(conf["train"]["num_workers"]),
        drop_last=True,
    )
    print("dataset speakers:", len(dataset))
    print("loader batches per pass:", len(loader))

    pn_model = load_frozen_pn_encoder(
        conf["paths"]["pn_checkpoint"],
        device,
        strict=bool(conf["pn_encoder"]["strict_load"]),
    )
    print("loaded frozen PN encoder:", conf["paths"]["pn_checkpoint"])

    cfm = _build_flowse(conf, device)
    model = PNConditionedCFM(cfm).to(device)
    trainable_params, optimizer, adapter_params, flowse_params = _configure_trainable(
        model, conf
    )
    print("adapter trainable params:", sum(p.numel() for p in adapter_params))
    print("flowse trainable params:", sum(p.numel() for p in flowse_params))
    print("total trainable params:", sum(p.numel() for p in trainable_params))

    if dry_run:
        print("dry-run complete")
        return

    save_dir = Path(conf["paths"]["save_dir"])
    writer = SummaryWriter(conf["paths"]["log_dir"])
    print("tensorboard log dir:", conf["paths"]["log_dir"])

    sample_rate = int(conf["data"]["sample_rate"])
    target_sr = int(conf["flowse"]["mel_spec"]["target_sample_rate"])
    if conf["flowse"]["init"] == "pretrained":
        flowse_config = _read_yaml(conf["paths"]["flowse_config"])
        target_sr = int(flowse_config["model"]["mel_spec"]["target_sample_rate"])

    resampler = None
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sr,
        ).to(device)

    active_num = list(conf["data"]["active_num"])
    steps_per_epoch = int(conf["train"]["steps_per_epoch"])
    epochs = int(conf["train"]["epochs"])
    log_interval = int(conf["train"]["log_interval"])
    save_every_epoch = int(conf["train"]["save_every_epoch"])
    clean_est_l1_weight = float(conf["flowse"]["clean_est_l1_weight"])
    grad_clip = float(conf["train"]["grad_clip"])

    global_step = 0
    best_loss = float("inf")
    model.train()

    try:
        for epoch in range(epochs):
            # Recreate the DataLoader iterator each epoch; do not use itertools.cycle(),
            # because cycle caches first-pass batches and can kill dataset randomness.
            iterator = iter(loader)
            total_loss = 0.0
            total_cfm = 0.0
            total_clean = 0.0
            count = 0

            for step in range(steps_per_epoch):
                try:
                    audio, pos, neg = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    audio, pos, neg = next(iterator)

                global_step += 1
                audio = audio.to(device)
                pos = pos.to(device)
                neg = neg.to(device)

                mix_wave = audio.sum(dim=1).squeeze(1)
                # Match pnenroll's target definition: only the active positive speaker.
                target_wave = audio[:, : active_num[1]].sum(dim=1).squeeze(1)

                with torch.no_grad():
                    cond_emb = pn_model.encode(pos.sum(dim=1), neg.sum(dim=1))
                    if resampler is not None:
                        mix_wave = resampler(mix_wave)
                        target_wave = resampler(target_wave)
                    noisy_mel = cfm.mel_spec(mix_wave).permute(0, 2, 1)
                    clean_mel = cfm.mel_spec(target_wave).permute(0, 2, 1)

                optimizer.zero_grad()
                cfm_loss, clean_l1, _pred, clean_est = _pn_flow_matching_step(
                    cfm,
                    noisy_mel,
                    clean_mel,
                    cond_emb,
                )
                loss = cfm_loss + clean_est_l1_weight * clean_l1

                if not torch.isfinite(loss):
                    print(
                        "non-finite loss at "
                        f"epoch={epoch + 1} step={step + 1}: {loss.item()}"
                    )
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                optimizer.step()

                total_loss += loss.item()
                total_cfm += cfm_loss.item()
                total_clean += clean_l1.item()
                count += 1

                if global_step == 1 or global_step % log_interval == 0:
                    with torch.no_grad():
                        pn_tokens = model.cfm.transformer.pn_projector(cond_emb)
                        pn_norm = pn_tokens.norm(dim=-1).mean().item()
                        gate = model.cfm.transformer.speaker_attn_gate.item()
                        clean_delta = (clean_est - clean_mel).abs().mean().item()

                    avg_loss = total_loss / max(count, 1)
                    print(
                        f"epoch {epoch + 1:03d} | step {global_step:06d} | "
                        f"loss {loss.item():.6f} | avg {avg_loss:.6f} | "
                        f"cfm {cfm_loss.item():.6f} | clean_l1 {clean_l1.item():.6f} | "
                        f"pn_norm {pn_norm:.6f} | gate {gate:.6f} | "
                        f"clean_delta {clean_delta:.6f}",
                        flush=True,
                    )
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar("train/cfm_loss", cfm_loss.item(), global_step)
                    writer.add_scalar("train/clean_l1", clean_l1.item(), global_step)
                    writer.add_scalar("train/pn_norm", pn_norm, global_step)
                    writer.add_scalar("train/gate", gate, global_step)

            metrics = {
                "loss": total_loss / max(count, 1),
                "cfm_loss": total_cfm / max(count, 1),
                "clean_l1": total_clean / max(count, 1),
            }
            print(f"epoch {epoch + 1:03d} done | {metrics}", flush=True)
            writer.add_scalar("epoch/loss", metrics["loss"], epoch + 1)
            writer.add_scalar("epoch/cfm_loss", metrics["cfm_loss"], epoch + 1)
            writer.add_scalar("epoch/clean_l1", metrics["clean_l1"], epoch + 1)
            writer.flush()

            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                _save_checkpoint(
                    save_dir / "best.pt",
                    model,
                    optimizer,
                    conf,
                    epoch + 1,
                    global_step,
                    metrics,
                )

            if save_every_epoch > 0 and (epoch + 1) % save_every_epoch == 0:
                _save_checkpoint(
                    save_dir / f"epoch_{epoch + 1:04d}.pt",
                    model,
                    optimizer,
                    conf,
                    epoch + 1,
                    global_step,
                    metrics,
                )
    finally:
        writer.close()
