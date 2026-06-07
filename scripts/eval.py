#!/usr/bin/env python
import logging
from argparse import ArgumentParser
from pathlib import Path

import torch

from sia_fm_tse.model import CFM, DiT, Encoder, FlowTSE
from sia_fm_tse.utils import EvalConf, WandbHandler, eval, load_eval_config


def main(handler: logging.Handler):
    # ===== Arguments ===== #
    parser = ArgumentParser("Evaluation for FlowTSE v1 model")
    parser.add_argument(
        "--config",
        help="path to eval config file",
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
    arguments = parser.parse_args()

    # ===== Logging ===== #
    # root defaults to WARNING, which would drop all logger.info() -> wandb.log()
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)

    # ===== Configs ===== #
    conf: EvalConf = load_eval_config(arguments.config)
    logger.info(f"config: {arguments.config}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    # ===== Model ===== #
    logger.info("loading encoder...")
    encoder = Encoder()
    encoder.load(conf.encoder_ckpt_path)
    encoder.eval()
    logger.info(f"encoder loaded from {conf.encoder_ckpt_path}")

    spec_dim = 2 * (conf.n_fft // 2 + 1)
    transformer = DiT(
        dim=conf.dim,
        depth=conf.depth,
        n_head=conf.n_head,
        dim_head=conf.dim_head,
        dropout=conf.dropout,
        ff_mult=conf.ff_mult,
        mel_dim=spec_dim,
        long_skip_connection=conf.long_skip_connection,
        cond_in_ch=conf.cond_in_ch,
        cond_in_freq=conf.cond_in_freq,
    )
    decoder = CFM(
        transformer=transformer,
        cond_drop_prob=conf.cond_drop_prob,
    )

    logger.info("loading decoder...")
    ckpt = torch.load(conf.decoder_ckpt_path, map_location=device)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()
    logger.info(f"decoder loaded from {conf.decoder_ckpt_path}")

    model = FlowTSE(
        encoder,
        decoder,
        n_fft=conf.n_fft,
        hop_length=conf.hop_length,
        win_length=conf.win_length,
    ).to(device)

    # ===== Evaluation ===== #
    results = eval(
        model,
        conf,
        data_dir=str(arguments.data_dir),
        noise_dir=str(arguments.noise_dir / "tt") + "/",
        handler=handler,
    )
    logger.info(f"results: {results}")


if __name__ == "__main__":
    main(handler=WandbHandler())
