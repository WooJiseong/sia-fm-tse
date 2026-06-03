from pathlib import Path

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
        frozen=True,
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


class ModelConfig(EncoderConfig, DecoderConfig): ...


class DataConfig(BaseModel):
    n_mels: int = Field(
        default=100,
        description="Number of mel filterbanks for data",
    )
    sample_rate: int = Field(
        default=16000,
        description="(Constant) Sample rate for whole model",
        frozen=True,
    )
    snr_db_range: tuple[int, int] = Field(
        default=(0, 0),
        description="SNR Range for random noise insertion",
    )
    mixture_speakers: int = Field(
        default=3,
        description="Number of speakers in audio mixture",
    )
    positive_enroll_speakers: int = Field(
        default=1,
        description="Number of speakers in positiive audio enrollment",
    )
    negative_enroll_speakers: int = Field(
        default=2,
        description="Number of speakers in negative audio enrollment",
    )
    batch_size: int = Field(
        default=4,
        description="Size of batch from dataset",
    )


class TrainConfig(BaseModel):
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


class Config(ModelConfig, DataConfig, TrainConfig): ...


def load_config(path: str) -> Config:
    with Path(path).open() as f:
        data = yaml.safe_load(f)

    return Config.model_validate(data, strict=True)
