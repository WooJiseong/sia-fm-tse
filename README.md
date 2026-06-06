# sia-fm-tse

Target Speaker Extraction via Positive/Negative Enrollment + Flow Matching decoder.

---

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (Recommended, but optional)
- CUDA 11.8 (matching the torch build)

---

## Installation

```bash
just setup
# or manually
uv sync --frozen && uv add . --dev --editable
```

If you don't use uv,

```bash
pip install -r requirements.txt
```

---

## Encoder Checkpoint

The encoder uses pretrained weights from `proposed-monaural.pt`.

- Source: [ShitongXu/TSE-Pos-Neg-Enroll](https://huggingface.co/ShitongXu/TSE-Pos-Neg-Enroll/blob/main/proposed-monaural.pt)
- Place the downloaded file at `checkpoints/encoder.pt`.

```
checkpoints/
└── encoder.pt
```

---

## Data

| Purpose | Path | Format |
|---------|------|--------|
| Speech mixtures | `data/LibriSpeech/train-clean-360` | LibriSpeech |
| Noise | `data/wham_noise/` | WHAM! noise dataset (`tr/`, `cv/`, `tt/` subdirs) |

---

## Config

Model, data, and training settings are managed in `configs/train.yaml`. See [`configs/train.yaml`](configs/train.yaml) for an example.

| Field | Key | Default |
|-------|-----|---------|
| Encoder checkpoint | `encoder_ckpt_path` | — (required) |
| DiT embedding dim | `dim` | — (required) |
| DiT depth | `depth` | `8` |
| Attention heads | `n_head` | `8` |
| Dropout | `dropout` | `0.1` |
| CFG drop probability | `cond_drop_prob` | `0.1` |
| Mel channels | `n_mels` | `100` |
| Speakers in mixture | `source_num` | `3` |
| Min speakers in mixture | `min_source_num` | `3` |
| Pos/neg enrollment split | `active_num` | `[-1, 1]` |
| Batch size | `batch_size` | `4` |
| Epochs | `epochs` | `10` |
| Steps per epoch | `steps_per_epoch` | `500` |
| Learning rate | `lr` | `1e-4` |
| Gradient clip norm | `grad_clip` | `1.0` |

---

## Training

```bash
uv run python scripts/train.py \
  --config configs/train.yaml \
  --data_dir data/LibriSpeech/train-clean-360 \
  --noise_dir data/wham_noise \
  --save_dir checkpoints/runs
```

The checkpoint is saved to `checkpoints/runs/flow_tse_crossattnv2_bs{batch_size}_epoch{epochs}.pt` when training finishes.

### W&B Logging

Runs are saved locally in offline mode by default (no login required). To upload a run after training:

```bash
wandb sync wandb/
```
