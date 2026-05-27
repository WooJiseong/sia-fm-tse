"""Train a TFGridNet encoder checkpoint compatible with pnenroll/train.py.

The paper README expects a LookOnceToHear-style TFGridNet checkpoint at
``model/best.ckpt``.  The training code then strips the leading ``"model."``
prefix from every key and loads the remaining parameters into
``model.tfgridnet_encoder.TFGridNet_encoder``.

This script trains that encoder from the local LibriSpeech/WHAM data by using a
small decoder head only for pretraining.  At save time only the TFGridNet encoder
state is exported, with the required ``"model."`` prefix, so the resulting
checkpoint can be used as the frozen "Pretrained TFGridNet Encoder".
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from model.tfgridnet_encoder import TFGridNet_encoder
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Failed to import TFGridNet_encoder. Install pnenroll/requirements.txt "
        "from this directory first; espnet is required for TFGridNet."
    ) from exc


BASE_DIR = Path(__file__).resolve().parent


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """Runtime options for TFGridNet encoder pretraining."""

    data_root: Path
    train_splits: Tuple[str, ...]
    val_splits: Tuple[str, ...]
    noise_train_dir: Optional[Path]
    noise_val_dir: Optional[Path]
    output_ckpt: Path
    model_copy: Optional[Path]
    sample_rate: int
    segment_seconds: float
    steps_per_epoch: int
    val_steps: int
    epochs: int
    batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    grad_clip: float
    source_num: int
    snr_min: float
    snr_max: float
    noise_snr_min: float
    noise_snr_max: float
    n_fft: int
    stride: int
    num_blocks: int
    emb_dim: int
    binaural: bool
    seed: int
    device: str

    @property
    def segment_length(self) -> int:
        """Number of waveform samples in one training segment."""

        return int(round(self.sample_rate * self.segment_seconds))


def resolve_path(path_text: Optional[str]) -> Optional[Path]:
    """Resolve CLI paths relative to this script directory."""

    if path_text is None or path_text == "":
        return None
    path = Path(path_text)
    return path if path.is_absolute() else BASE_DIR / path


def parse_args() -> TrainConfig:
    """Parse command line arguments into a typed configuration object."""

    parser = argparse.ArgumentParser(
        description=(
            "Pretrain a TFGridNet encoder and save a best.ckpt compatible with "
            "sia_fm_tse/pnenroll/train.py."
        )
    )
    parser.add_argument("--data-root", default="data/LibriSpeech")
    parser.add_argument("--train-splits", nargs="+", default=["train-clean-360"])
    parser.add_argument("--val-splits", nargs="+", default=["dev-clean"])
    parser.add_argument("--noise-train-dir", default="data/wham_noise/tr")
    parser.add_argument("--noise-val-dir", default="data/wham_noise/cv")
    parser.add_argument("--output-ckpt", default="runs/embed/best.ckpt")
    parser.add_argument(
        "--model-copy",
        default="model/best.ckpt",
        help=(
            "Optional extra copy path. The existing train.py loads model/best.ckpt, "
            "while the README link historically used runs/embed/best.ckpt."
        ),
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--val-steps", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--source-num", type=int, default=2)
    parser.add_argument("--snr-min", type=float, default=-2.5)
    parser.add_argument("--snr-max", type=float, default=2.5)
    parser.add_argument("--noise-snr-min", type=float, default=5.0)
    parser.add_argument("--noise-snr-max", type=float, default=20.0)
    parser.add_argument("--n-fft", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--binaural", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Examples: cuda, cuda:0, cpu.",
    )
    args = parser.parse_args()

    return TrainConfig(
        data_root=resolve_path(args.data_root) or BASE_DIR / "data/LibriSpeech",
        train_splits=tuple(args.train_splits),
        val_splits=tuple(args.val_splits),
        noise_train_dir=resolve_path(args.noise_train_dir),
        noise_val_dir=resolve_path(args.noise_val_dir),
        output_ckpt=resolve_path(args.output_ckpt) or BASE_DIR / "runs/embed/best.ckpt",
        model_copy=resolve_path(args.model_copy),
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        steps_per_epoch=args.steps_per_epoch,
        val_steps=args.val_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        source_num=args.source_num,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        noise_snr_min=args.noise_snr_min,
        noise_snr_max=args.noise_snr_max,
        n_fft=args.n_fft,
        stride=args.stride,
        num_blocks=args.num_blocks,
        emb_dim=args.emb_dim,
        binaural=args.binaural,
        seed=args.seed,
        device=args.device,
    )


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch RNGs for reproducible sampling."""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_audio_files(root: Path, splits: Sequence[str]) -> Dict[str, List[Path]]:
    """Return LibriSpeech audio files grouped by speaker id."""

    speaker_files: Dict[str, List[Path]] = {}
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"LibriSpeech split not found: {split_dir}")

        for flac_path in split_dir.glob("*/*/*.flac"):
            speaker_id = flac_path.parts[-3]
            speaker_files.setdefault(speaker_id, []).append(flac_path)

    speaker_files = {
        speaker_id: sorted(paths)
        for speaker_id, paths in sorted(speaker_files.items())
        if paths
    }
    if len(speaker_files) < 2:
        raise ValueError(
            f"Need at least two speakers under {root} splits {splits}; "
            f"found {len(speaker_files)}."
        )
    return speaker_files


def list_noise_files(noise_dir: Optional[Path]) -> List[Path]:
    """Return optional WHAM noise files for additive augmentation."""

    if noise_dir is None or not noise_dir.is_dir():
        return []
    return sorted(noise_dir.glob("*.wav"))


def rms(audio: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute waveform RMS with a small floor for numerical stability."""

    return torch.sqrt(audio.pow(2).mean().clamp_min(eps))


def match_rms(source: torch.Tensor, reference: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Scale source so it has the requested SNR relative to reference."""

    desired = rms(reference) / math.sqrt(10.0 ** (snr_db / 10.0))
    return source * (desired / rms(source))


class LibriSpeechMixtureDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Randomly synthesize target-speaker mixtures from LibriSpeech files."""

    def __init__(
        self,
        speaker_files: Dict[str, List[Path]],
        noise_files: Sequence[Path],
        sample_rate: int,
        segment_length: int,
        steps: int,
        source_num: int,
        snr_range: Tuple[float, float],
        noise_snr_range: Tuple[float, float],
        seed: int,
    ) -> None:
        if source_num < 1:
            raise ValueError("source_num must be >= 1")
        self.speaker_files = speaker_files
        self.speaker_ids = sorted(speaker_files)
        self.noise_files = list(noise_files)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.steps = steps
        self.source_num = source_num
        self.snr_range = snr_range
        self.noise_snr_range = noise_snr_range
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        """Return configured random samples per epoch."""

        return self.steps

    def set_epoch(self, epoch: int) -> None:
        """Change the deterministic random stream between training epochs."""

        self.epoch = epoch

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create one mixture and its clean target source."""

        rng = random.Random(self.seed + self.epoch * self.steps + index)
        target_speaker = rng.choice(self.speaker_ids)
        interferer_pool = [speaker for speaker in self.speaker_ids if speaker != target_speaker]
        interferer_speakers = rng.sample(interferer_pool, self.source_num - 1)

        target = self._load_random_segment(target_speaker, rng)
        sources = [target]
        for speaker_id in interferer_speakers:
            interferer = self._load_random_segment(speaker_id, rng)
            snr_db = rng.uniform(*self.snr_range)
            sources.append(match_rms(interferer, target, snr_db))

        mixture = torch.stack(sources, dim=0).sum(dim=0)
        if self.noise_files:
            noise = self._load_noise(rng)
            snr_db = rng.uniform(*self.noise_snr_range)
            mixture = mixture + match_rms(noise, target, snr_db)

        peak = mixture.abs().max().clamp_min(1.0)
        mixture = mixture / peak
        target = target / peak
        return mixture.unsqueeze(0), target.unsqueeze(0)

    def _load_random_segment(self, speaker_id: str, rng: random.Random) -> torch.Tensor:
        """Load and concatenate utterances until a fixed segment is available."""

        chunks: List[torch.Tensor] = []
        total_length = 0
        while total_length < self.segment_length:
            path = rng.choice(self.speaker_files[speaker_id])
            audio = self._load_audio(path)
            chunks.append(audio)
            total_length += audio.numel()

        audio = torch.cat(chunks, dim=-1)
        if audio.numel() > self.segment_length:
            max_start = audio.numel() - self.segment_length
            start = rng.randint(0, max_start)
            audio = audio[start : start + self.segment_length]
        return audio

    def _load_noise(self, rng: random.Random) -> torch.Tensor:
        """Load a WHAM noise segment with repeat padding if it is too short."""

        noise = self._load_audio(rng.choice(self.noise_files))
        if noise.numel() < self.segment_length:
            repeats = math.ceil(self.segment_length / noise.numel())
            noise = noise.repeat(repeats)
        max_start = noise.numel() - self.segment_length
        start = rng.randint(0, max_start) if max_start > 0 else 0
        return noise[start : start + self.segment_length]

    def _load_audio(self, path: Path) -> torch.Tensor:
        """Load mono audio and resample to the configured sample rate."""

        audio, sr = torchaudio.load(str(path))
        audio = audio.mean(dim=0)
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        return audio.float()


class TFGridNetPretrainer(nn.Module):
    """TFGridNet encoder with a temporary decoder head for waveform pretraining."""

    def __init__(
        self,
        n_fft: int,
        stride: int,
        num_blocks: int,
        emb_dim: int,
        binaural: bool,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.stride = stride
        self.binaural = binaural
        self.output_channels = 2 if binaural else 1
        self.encoder = TFGridNet_encoder(
            num_ch=2,
            n_fft=n_fft,
            stride=stride,
            num_blocks=num_blocks,
            binaural=binaural,
        )
        self.decoder = nn.ConvTranspose2d(
            emb_dim,
            self.output_channels * 2,
            kernel_size=(3, 3),
            padding=(0, 1),
        )

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        """Estimate the clean target waveform from a mixture.

        Args:
            mixture: Tensor shaped ``[batch, channels, samples]``.

        Returns:
            Tensor shaped ``[batch, output_channels, samples]``.
        """

        length = mixture.shape[-1]
        if self.binaural and mixture.shape[1] == 1:
            mixture = mixture.repeat(1, 2, 1)
        encoder_input = mixture.transpose(1, 2)
        input_std = encoder_input.std(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        encoded = self.encoder(encoder_input, None)
        decoded = self.decoder(encoded)
        return self._istft(decoded, length, input_std)

    def _istft(
        self,
        decoded: torch.Tensor,
        length: int,
        input_std: torch.Tensor,
    ) -> torch.Tensor:
        """Convert decoder real/imaginary bins back to waveform audio."""

        batch_size, _, _, _ = decoded.shape
        decoded = decoded.view(
            batch_size,
            self.output_channels,
            2,
            decoded.shape[-2],
            decoded.shape[-1],
        )
        real = decoded[:, :, 0].transpose(2, 3).contiguous()
        imag = decoded[:, :, 1].transpose(2, 3).contiguous()
        complex_spec = torch.complex(real, imag)

        flat_spec = complex_spec.reshape(-1, complex_spec.shape[-2], complex_spec.shape[-1])
        waveform = torch.istft(
            flat_spec,
            n_fft=self.n_fft,
            hop_length=self.stride,
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft, device=decoded.device),
            length=length,
        )
        waveform = waveform.view(batch_size, self.output_channels, length)
        return waveform * input_std.transpose(1, 2)


def si_snr_loss(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return negative SI-SNR, averaged over batch and channels."""

    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    projection = (estimate * target).sum(dim=-1, keepdim=True) * target
    projection = projection / target.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps)
    noise = estimate - projection
    ratio = projection.pow(2).sum(dim=-1) / noise.pow(2).sum(dim=-1).clamp_min(eps)
    return -10.0 * torch.log10(ratio.clamp_min(eps)).mean()


def make_loader(
    speaker_files: Dict[str, List[Path]],
    noise_files: Sequence[Path],
    config: TrainConfig,
    steps: int,
    seed: int,
    shuffle: bool,
) -> DataLoader[Tuple[torch.Tensor, torch.Tensor]]:
    """Construct a random-mixture dataloader."""

    dataset = LibriSpeechMixtureDataset(
        speaker_files=speaker_files,
        noise_files=noise_files,
        sample_rate=config.sample_rate,
        segment_length=config.segment_length,
        steps=steps,
        source_num=config.source_num,
        snr_range=(config.snr_min, config.snr_max),
        noise_snr_range=(config.noise_snr_min, config.noise_snr_max),
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available() and config.device.startswith("cuda"),
        drop_last=False,
    )


def train_one_epoch(
    model: TFGridNetPretrainer,
    loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    """Run one optimization epoch and return mean loss."""

    model.train()
    losses: List[float] = []
    for mixture, target in tqdm(loader, desc="train", unit="batch"):
        mixture = mixture.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        estimate = model(mixture)
        target = match_target_channels(target, estimate.shape[1])
        loss = si_snr_loss(estimate, target)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def validate(
    model: TFGridNetPretrainer,
    loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> float:
    """Evaluate mean validation loss."""

    model.eval()
    losses: List[float] = []
    for mixture, target in tqdm(loader, desc="val", unit="batch"):
        mixture = mixture.to(device)
        target = target.to(device)
        estimate = model(mixture)
        target = match_target_channels(target, estimate.shape[1])
        losses.append(float(si_snr_loss(estimate, target).cpu()))
    return sum(losses) / max(len(losses), 1)


def match_target_channels(target: torch.Tensor, channels: int) -> torch.Tensor:
    """Repeat mono targets when a binaural encoder is being pretrained."""

    if target.shape[1] == channels:
        return target
    if target.shape[1] == 1:
        return target.repeat(1, channels, 1)
    raise ValueError(f"Cannot map target with {target.shape[1]} channels to {channels}.")


def prefixed_encoder_state_dict(model: TFGridNetPretrainer) -> Dict[str, torch.Tensor]:
    """Return encoder weights in the ``best.ckpt`` format expected by train.py."""

    return {
        f"model.{key}": value.detach().cpu()
        for key, value in model.encoder.state_dict().items()
    }


def save_best_ckpt(
    model: TFGridNetPretrainer,
    config: TrainConfig,
    epoch: int,
    val_loss: float,
    path: Path,
) -> None:
    """Save only the pretrained encoder using the expected checkpoint schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": prefixed_encoder_state_dict(model),
        "epoch": epoch,
        "val_loss": val_loss,
        "meta": {
            "format": "pnenroll_tfgridnet_encoder_prefixed",
            "prefix": "model.",
            "loader_note": "pnenroll/train.py strips key[:6] before loading.",
            "config": prepare_jsonable_config(config),
        },
    }
    torch.save(payload, path)


def write_direct_encoder_checkpoint(model: TFGridNetPretrainer, output_ckpt: Path) -> None:
    """Save an additional plain encoder state_dict for debugging or fine-tuning."""

    plain_path = output_ckpt.with_name(output_ckpt.stem + "_encoder.pt")
    torch.save({"state_dict": model.encoder.state_dict()}, plain_path)


def copy_for_train_py(source: Path, destination: Optional[Path]) -> None:
    """Copy best.ckpt to model/best.ckpt when requested."""

    if destination is None:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def prepare_jsonable_config(config: TrainConfig) -> Dict[str, object]:
    """Convert dataclass config paths to strings for logging."""

    config_dict = dataclasses.asdict(config)
    for key, value in list(config_dict.items()):
        if isinstance(value, Path):
            config_dict[key] = str(value)
    return config_dict


def main() -> None:
    """Entry point for TFGridNet encoder pretraining."""

    config = parse_args()
    seed_everything(config.seed)
    device = torch.device(config.device)

    print(json.dumps(prepare_jsonable_config(config), indent=2, sort_keys=True))

    train_speakers = list_audio_files(config.data_root, config.train_splits)
    val_speakers = list_audio_files(config.data_root, config.val_splits)
    train_noise = list_noise_files(config.noise_train_dir)
    val_noise = list_noise_files(config.noise_val_dir)
    print(
        f"train speakers={len(train_speakers)}, val speakers={len(val_speakers)}, "
        f"train noise={len(train_noise)}, val noise={len(val_noise)}"
    )

    train_loader = make_loader(
        train_speakers,
        train_noise,
        config,
        steps=config.steps_per_epoch,
        seed=config.seed,
        shuffle=True,
    )
    val_loader = make_loader(
        val_speakers,
        val_noise,
        config,
        steps=config.val_steps,
        seed=config.seed + 10_000_000,
        shuffle=False,
    )

    model = TFGridNetPretrainer(
        n_fft=config.n_fft,
        stride=config.stride,
        num_blocks=config.num_blocks,
        emb_dim=config.emb_dim,
        binaural=config.binaural,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    best_val = float("inf")
    last_path = config.output_ckpt.with_name("last.ckpt")
    for epoch in range(config.epochs):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            grad_clip=config.grad_clip,
        )
        val_loss = validate(model, val_loader, device=device)
        scheduler.step(val_loss)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} lr={optimizer.param_groups[0]['lr']:.3e}"
        )

        save_best_ckpt(model, config, epoch, val_loss, last_path)
        if val_loss < best_val:
            best_val = val_loss
            save_best_ckpt(model, config, epoch, val_loss, config.output_ckpt)
            write_direct_encoder_checkpoint(model, config.output_ckpt)
            copy_for_train_py(config.output_ckpt, config.model_copy)
            print(f"saved best checkpoint to {config.output_ckpt}")

    print(f"finished; best_val_loss={best_val:.4f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted.")
