#!/usr/bin/env python
import logging
from argparse import ArgumentParser
from itertools import cycle, product
from pathlib import Path

import torch
import torchaudio
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from sia_fm_tse.model import CFM, DiT, Encoder, FlowTSE
from sia_fm_tse.utils import (
    LibriDataset,
    WandbHandler,
    get_vocos_mel_spectrogram,
    load_config,
)


def train(handler: logging.Handler):
    # ===== Arguments ===== #
    parser = ArgumentParser("Trainer for FlowTSE v1 model")
    parser.add_argument(
        "--config",
        help="path to config file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data_dir",
        help="path to LibriDataset directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--noise_dir",
        help="path to Wham-noise Dataset directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--save_dir",
        help="path to save FlowTSE checkpoint",
        type=Path,
        required=True,
    )
    arguments = parser.parse_args()

    # ===== Logging ===== #
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)

    # ===== Configs ===== #
    conf = load_config(arguments.config)
    logger.info(f"config: {arguments.config}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    # ===== Model ===== #
    logger.info("loading encoder...")
    encoder = Encoder()
    encoder.load(conf.encoder_ckpt_path)
    encoder.eval()
    logger.info(f"encoder successfully loaded from {conf.encoder_ckpt_path}")

    resampler = torchaudio.transforms.Resample(
        orig_freq=16000,
        new_freq=24000,
    ).to(device)

    transformer = DiT(
        dim=conf.dim,
        depth=conf.depth,
        n_head=conf.n_head,
        dim_head=conf.dim_head,
        dropout=conf.dropout,
        ff_mult=conf.ff_mult,
        mel_dim=conf.n_mels,
        long_skip_connection=conf.long_skip_connection,
        cond_in_ch=conf.cond_in_ch,
        cond_in_freq=conf.cond_in_freq,
    )

    decoder = CFM(
        transformer=transformer,
        cond_drop_prob=conf.cond_drop_prob,
    )

    model = FlowTSE(encoder, decoder)
    model.to(device)

    # ===== Dataset ===== #
    logger.info("loading dataset...")
    dataset = LibriDataset(
        str(arguments.data_dir),
        sample_rate=conf.sample_rate,
        wave_length=3 * conf.sample_rate,
        pos_example_length=3 * conf.sample_rate,
        neg_example_length=3 * conf.sample_rate,
        snr_db_range=conf.snr_db_range,
        min_source_num=conf.min_source_num,
        source_num=conf.source_num,
        active_num=conf.active_num,
        reproducable=False,
        normalize=False,
        filling_pattern="repeat",
        return_dvec=False,
        dvec_rate=50,
        include_silent=False,
        special_spk=[],
        reverb="none",
        binaural=False,
        reverb_cond=False,
        zero_in_tgt=False,
        noise_dir=str(arguments.noise_dir / "tr") + "/",
        same_disturb=False,
    )
    loader: DataLoader[tuple[torch.Tensor, ...]] = DataLoader(
        dataset,
        batch_size=conf.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    logger.info(
        "dataset loaded from "
        f"data_dir: {arguments.data_dir}, noise_dir: {arguments.noise_dir}"
    )

    # ===== Train ===== #
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=conf.lr)
    total_loss = 0
    loss_tractable = 0
    for (epoch, step), (audio, pos, neg) in tqdm(
        zip(
            product(range(conf.epochs), range(conf.steps_per_epoch)),
            cycle(loader),
            strict=True,
        )
    ):
        global_step = conf.steps_per_epoch * epoch + step

        audio: torch.Tensor = audio.to(device)
        pos: torch.Tensor = pos.to(device)
        neg: torch.Tensor = neg.to(device)

        mixture = audio.sum(dim=1).squeeze(1)
        target = audio[:, : conf.active_num[1]].sum(dim=1).squeeze(1)

        with torch.no_grad():
            condition = encoder(pos, neg)      # [B, C, T_enc, F] — passed directly

            # Resample and convert to mel spectrogram
            mixture = resampler(mixture)
            target = resampler(target)
            noise = get_vocos_mel_spectrogram(mixture, n_mel_channels=conf.n_mels)
            clean = get_vocos_mel_spectrogram(target, n_mel_channels=conf.n_mels)
            noise = rearrange(noise, "b d n -> b n d")
            clean = rearrange(clean, "b d n -> b n d")

        optimizer.zero_grad()
        loss = model.decoder.loss(noise, condition, clean)

        if not torch.isfinite(loss):
            logger.warning(
                f"non-finite loss at epoch {epoch + 1}, step {step + 1}: {loss.item()}"
            )
            continue
        else:
            loss_tractable += 1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), conf.grad_clip)
        optimizer.step()

        total_loss += loss.item()

        # Record
        if global_step == 0 or global_step % 20 == 0:
            with torch.no_grad():
                ...

            logger.info(
                f"epoch {epoch + 1:03d} | step {global_step:06d} | "
                f"loss {loss.item():.6f} | avg_loss {total_loss / (global_step + 1)}"
            )

    final_loss = total_loss / (global_step + 1)
    logger.info(f"All epoch ended ({conf.epochs}), with final epoch loss {final_loss}")

    total_steps = conf.steps_per_epoch * conf.epochs
    non_finite_steps = total_steps - loss_tractable
    logger.info(
        f"{non_finite_steps} / {total_steps} steps were terminated because of "
        f"non-finite loss ({non_finite_steps / total_steps:.2%})"
    )

    arguments.save_dir.mkdir(parents=True, exist_ok=True)
    save_path = (
        arguments.save_dir / f"flow_tse_crossattnv2_bs{conf.batch_size}_epoch{conf.epochs}.pt"
    )

    torch.save(
        {
            "epochs": conf.epochs,
            "steps": conf.steps_per_epoch * conf.epochs,
            "steps_per_epoch": conf.steps_per_epoch,
            "decoder": model.decoder.state_dict(),
            "final_loss": final_loss,
            "batch_size": conf.batch_size,
            "lr": conf.lr,
        },
        save_path,
    )
    logger.info(f"experiment saved at: {save_path}")
    logger.info("FlowTSE training has done")


if __name__ == "__main__":
    train(handler=WandbHandler())
