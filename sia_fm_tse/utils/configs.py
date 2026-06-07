from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, FilePath


class EncoderConfig(BaseModel):
    encoder_ckpt_path: FilePath = Field(
        ...,
        description="`proposed_monaural.pt` checkpoint path. See https://huggingface.co/ShitongXu/TSE-Pos-Neg-Enroll/blob/main/proposed-monaural.pt",
    )


class DiTBlockConfig(BaseModel):
    n_head: int = Field(
        default=8,
        description="Number of heads for attention in DiT Block",
    )
    dim_head: int = Field(
        default=64,
        description="Dimension of each Q, K, V Tensor for attention in DiT Block",
    )
    ff_mult: int = Field(
        default=4,
        description="Dimension multiplier of FFN in DiT Block",
    )


class DecoderConfig(DiTBlockConfig):
    dim: int = Field(
        ...,
        description="Dimension of embedded tensor for DiT, so called D",
    )
    depth: int = Field(
        default=8,
        description="Number of DiT Block consisting DiT",
    )
    dropout: float = Field(
        default=0.1,
        description="Dropout probability for DiT",
    )
    long_skip_connection: bool = Field(
        default=False,
        description="Skip-connection from x to post-DiT-Block for DiT",
    )
    cond_drop_prob: float = Field(
        default=0, description="CFM condition drop probability for DiT"
    )
    cond_in_ch: Literal[64] = Field(
        default=64,
        description="Channel dim of raw encoder output — fixed by pretrained encoder",
    )
    cond_in_freq: Literal[65] = Field(
        default=65,
        description="Frequency dim of raw encoder output — fixed by pretrained encoder",
    )


class ModelConfig(EncoderConfig, DecoderConfig): ...


class DataConfig(BaseModel):
    n_fft: int = Field(
        default=512,
        description="FFT size for STFT features",
    )
    hop_length: int = Field(
        default=128,
        description="Hop length for STFT features",
    )
    win_length: int = Field(
        default=512,
        description="Window length for STFT features",
    )
    sample_rate: Literal[16000] = Field(
        default=16000,
        description="Sample rate for whole model — fixed by pretrained encoder",
    )
    snr_db_range: tuple[int, int] = Field(
        default=(0, 0),
        description="SNR Range for random noise insertion",
    )
    source_num: int = Field(
        default=3,
        description="Number of speakers in audio mixture",
    )
    min_source_num: int = Field(
        default=3,
        description="Minimum number of speakers in audio mixture",
    )
    active_num: list[int] = Field(
        default=[-1, 1],
        description=(
            "[_, pos_active] — number of positively enrolled speakers (including target). "
            "Speakers in audio[:, :pos_active] are the extraction target; "
            "enroll_noise_pids[pos_active-1:] become hard-negative (overlap) enrollees. "
            "Set to [-1, 1] for single-target extraction."
        ),
    )


class TrainConfig(BaseModel):
    batch_size: int = Field(
        default=4,
        description="Size of batch from dataset",
    )
    epochs: int = Field(
        default=10,
        description="Number of epochs for training",
    )
    steps_per_epoch: int = Field(
        default=500,
        description="Number of steps per epoch for training",
    )
    grad_clip: float = Field(
        default=1.0,
        description="Max-norm to clip gradient norms for training",
    )
    lr: float = Field(
        default=1e-4,
        description="Learning rate for training",
    )


class EvalConf(ModelConfig, DataConfig):
    decoder_ckpt_path: FilePath = Field(
        ...,
        description="Trained `decoder.pt` checkpoint path.",
    )


class TrainConf(ModelConfig, DataConfig, TrainConfig): ...


def load_eval_config(path: str) -> EvalConf:
    with Path(path).open() as f:
        data = yaml.safe_load(f)
    return EvalConf.model_validate(data, strict=False)


def load_config(path: str) -> TrainConf:
    with Path(path).open() as f:
        data = yaml.safe_load(f)
    return TrainConf.model_validate(data, strict=False)
