#!/usr/bin/env python
import logging
from argparse import ArgumentParser

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
        type=str,
        required=True,
    )
    parser.add_argument(
        "--noise_dir",
        help="path to Wham-noise Dataset directory",
        type=str,
        required=True,
    )
    arguments = parser.parse_args()

    # ===== Logging ===== #
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

    transformer = DiT(
        dim=conf.dim,
        depth=conf.depth,
        n_head=conf.n_head,
        dim_head=conf.dim_head,
        dropout=conf.dropout,
        ff_mult=conf.ff_mult,
        mel_dim=conf.n_mels,
        long_skip_connection=conf.long_skip_connection,
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

    model = FlowTSE(encoder, decoder).to(device)

    # ===== Evaluation ===== #
    results = eval(
        model,
        conf,
        data_dir=arguments.data_dir,
        noise_dir=arguments.noise_dir,
        handler=handler,
    )
    logger.info(f"results: {results}")


if __name__ == "__main__":
    main(handler=WandbHandler())
