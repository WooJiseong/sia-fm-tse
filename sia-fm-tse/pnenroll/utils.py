from dataclasses import dataclass
from typing import Literal

import yaml


@dataclass(frozen=True)
class Config:
    gpu: str
    batch_size: int
    epoch_num: int
    encoder_lr: float
    main_lr: float
    lr_schedule: str
    mode: Literal["min", "max"]
    patience: int
    factor: float
    min_lr: float
    save_epoch: int
    optimizer: float
    layer_num: int
    fusion_name: str
    pooling_size: int
    fusion_stride: int
    model_name: str
    load_model: str
    train_dataset_dir: str
    val_dataset_dir: str
    sample_rate: int
    wave_length: int
    pos_example_length: int
    neg_example_length: int
    dvec_rate: int
    return_clean_dvec: bool
    snr_db_range: list[int]
    source_num: int
    min_source_num: int
    active_num: list[int]
    reproducable: bool
    normalize: bool
    filling_pattern: str
    reverb: str
    brir_dir: list[str]
    binaural: bool
    zero_degree_pos: bool
    concat_pos: bool
    reverb_cond: bool
    zero_in_tgt: bool
    noise_dir: str
    special_spk: list[str]
    PI_range: list[float]
    NI_range: list[float]
    tgt_snr: int
    same_disturb: bool
    SNR_Weight: float
    Embedding_Weight: float
    emb_dim: int
    n_layers: int
    gradient_clip_val: float | None = None
    loss_type: str | None = None
    perturb_speeds: None = None
    emb_loss_type: str | None = None
    frozen_encoder: str | None = None
    DEC_Weight: float | None = None
    head_lr: float | None = None
    load_encoder: str | None = None
    weight_decay: float | None = None
    load_conv: bool | None = None
    freeze_conv: bool | None = None
    hidden_channels: int | None = None
    n_fft: int | None = None
    stride: int | None = None
    lstm_hidden_units: int | None = None
    emb_ks: int | None = None
    model_normalize: bool | None = None
    fusion_layer: list[int] | None = None
    refine_layer_num: int | None = None
    fusion_shortcut: list[int] | None = None
    cut_pos: bool | None = None
    sep_layer_num: int | None = None
    out_dim: int | None = None
    load_encoder_head: str | None = None
    enc_num_block: int | None = None
    lr_decay_epoch: int | None = None
    lr_decay_gamma: float | None = None


def get_config(yaml_config_filename: str) -> Config:
    with open(yaml_config_filename) as f:
        config_dict: dict = yaml.safe_load(f)

    return Config(**config_dict)
