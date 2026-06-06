import logging
from typing import Any

import torch
import torch.nn as nn
from speechmos import dnsmos as _dnsmos
from torch.utils.data import DataLoader
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio, SignalNoiseRatio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.metric import Metric
from tqdm import tqdm

from . import LibriDataset
from .configs import EvalConf


class DeepNoiseSuppressionMeanOpinionScore(Metric):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.add_state("ovrl_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor) -> None:
        audio = pred.squeeze().float().cpu().numpy()
        result = _dnsmos.run(audio, sr=16000)
        self.ovrl_sum += result["ovrl_mos"]  # type: ignore
        self.total += 1

    def compute(self) -> torch.Tensor:
        return self.ovrl_sum / self.total  # type: ignore


@torch.no_grad()
def eval(
    model: nn.Module,
    conf: EvalConf,
    *,
    data_dir: str,
    noise_dir: str,
    handler: logging.Handler,
) -> dict[str, float]:
    """
    Evaluate model with PESQ, SiSNR, SNR, STOI, DNSMOS.
    model should return like: tuple(pred, _)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== Logging ===== #
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)

    # ===== Metrics ===== #
    pesq = PerceptualEvaluationSpeechQuality(fs=16000, mode="wb")
    sisnr = ScaleInvariantSignalNoiseRatio()
    snr = SignalNoiseRatio()
    stoi = ShortTimeObjectiveIntelligibility(fs=16000)
    dnsmos = DeepNoiseSuppressionMeanOpinionScore()

    # ===== Dataset ===== #
    dataset = LibriDataset(
        data_dir,
        sample_rate=conf.sample_rate,
        wave_length=3 * conf.sample_rate,
        pos_example_length=3 * conf.sample_rate,
        neg_example_length=3 * conf.sample_rate,
        snr_db_range=conf.snr_db_range,
        min_source_num=conf.min_source_num,
        source_num=conf.source_num,
        active_num=conf.active_num,
        reproducable=True,
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
        noise_dir=noise_dir,
        same_disturb=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
    )

    logger.info("Starting evaluation...")

    for audio, pos, neg in tqdm(dataloader):
        audio: torch.Tensor = audio.to(device)
        pos: torch.Tensor = pos.to(device)
        neg: torch.Tensor = neg.to(device)
        mixture = audio.sum(dim=1)
        target = audio[:, : conf.active_num[1]].sum(dim=1)
        pred, _ = model(mixture, pos, neg)
        pesq.update(pred, target)
        snr.update(pred, target)
        sisnr.update(pred, target)
        stoi.update(pred, target)
        dnsmos.update(pred)

    results = {
        "pesq": pesq.compute().item(),
        "snr": snr.compute().item(),
        "si_snr": sisnr.compute().item(),
        "stoi": stoi.compute().item(),
        "dnsmos": dnsmos.compute().item(),
    }

    logger.info(results)
    return results
