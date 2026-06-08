# Mask2Flow Phase A Notes

This workspace adds a non-invasive Mask2Flow Phase A path on top of the latest
`WooJiseong/sia-fm-tse` CAv2 branch.

## What changed

- Added `Mask2FlowDiT`, a DiT refiner that receives:
  - flow state `xt`
  - original mixture STFT `Y`
  - PN encoder condition `c`
- Added `Mask2FlowCFM`, which learns:
  - `x0 = S0` coarse target estimate
  - `x1 = S` clean target STFT
  - `v = S - S0`
- Added cache and train scripts:
  - `scripts/build_mask2flow_cache.py`
  - `scripts/train_mask2flow_refiner.py`
- Added config:
  - `configs/mask2flow_refiner.yaml`

## Build cache

```powershell
C:\Users\mirac\anaconda3\python.exe scripts\build_mask2flow_cache.py `
  --config configs\train.yaml `
  --data_dir PATH\TO\LibriSpeech\train-clean-360 `
  --noise_dir PATH\TO\wham_noise `
  --cache_dir outputs\mask2flow_cache\oracle_irm `
  --coarse_mode oracle_irm `
  --num_samples 1000 `
  --batch_size 4
```

Useful `--coarse_mode` values:

- `mixture`: sanity check, should be close to CAv2 start point.
- `oracle_irm`: practical upper-bound coarse mask using mixture phase.
- `oracle_complex`: perfect coarse estimate, only for debugging.

## Dry run train

```powershell
C:\Users\mirac\anaconda3\python.exe scripts\train_mask2flow_refiner.py `
  --config configs\mask2flow_refiner.yaml `
  --dry-run
```

## Train

```powershell
C:\Users\mirac\anaconda3\python.exe scripts\train_mask2flow_refiner.py `
  --config configs\mask2flow_refiner.yaml
```

## First experiment table

Compare these under the same CAv2 STFT config, data split, PN encoder checkpoint,
and evaluator:

| Start point | Purpose |
| --- | --- |
| `randn -> source` | old baseline |
| `mixture -> source` | current CAv2 |
| `oracle_irm -> source` | Mask2Flow upper-bound-ish coarse start |
| `PN baseline S0 -> source` | real Phase A target |
| `learnable MaskNet S0 -> source` | Phase B |
