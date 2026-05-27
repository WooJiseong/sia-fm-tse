#!/bin/bash
#SBATCH --job-name=tfgridnet_pretrain
#SBATCH --output=logs/tfgridnet_%j.out
#SBATCH --error=logs/tfgridnet_%j.err

#SBATCH --partition=gpu4
#SBATCH --gres=gpu:2

#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G

#SBATCH --time=24:00:00

echo "======================================"
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "======================================"

# 프로젝트 루트로 이동
cd ~/SIALab/sia-fm-tse || exit 1

# 가상환경 활성화
source .venv/bin/activate

echo "Python path: $(which python)"
python --version

# 로그 디렉토리 생성
mkdir -p logs
mkdir -p runs/embed
mkdir -p model

echo "======================================"
echo "Starting TFGridNet pretraining..."
echo "======================================"

python sia_fm_tse/pnenroll/train_pre_tfgridnet.py \
    --data-root data/LibriSpeech \
    --train-splits train-clean-360 \
    --val-splits dev-clean \
    --noise-train-dir data/wham_noise/tr \
    --noise-val-dir data/wham_noise/cv \
    --output-ckpt runs/embed/best.ckpt \
    --model-copy model/best.ckpt \
    --epochs 100 \
    --batch-size 4 \
    --num-workers 8 \
    --lr 5e-4 \
    --device cuda

EXIT_CODE=$?

echo "======================================"
echo "Job finished at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "======================================"

exit ${EXIT_CODE}