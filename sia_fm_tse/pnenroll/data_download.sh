#!/bin/bash
#SBATCH --job-name=data_download
#SBATCH -p cpu2
#SBATCH --cpus-per-task=2
#SBATCH --output=download_%j.log

set -e

BASE_DATA_DIR="./data"
mkdir -p "$BASE_DATA_DIR"

echo "=================================================="
echo "Starting data download and setup in $BASE_DATA_DIR"
echo "=================================================="

# --------------------------------------------------
# 1. LibriSpeech Dataset (train-clean-360, dev-clean, test-clean)
# --------------------------------------------------

LIBRISPEECH_DIR="$BASE_DATA_DIR/LibriSpeech"
mkdir -p "$LIBRISPEECH_DIR"
echo "[1/2] Downloading LibriSpeech splits..."

LIBRISPEECH_SPLITS=("train-clean-360" "dev-clean" "test-clean")
for split in "${LIBRISPEECH_SPLITS[@]}"; do
    if [ ! -d "$LIBRISPEECH_DIR/$split" ]; then
        echo "Downloading $split..."
        wget -q --show-progress http://www.openslr.org/resources/12/$split.tar.gz -O "$BASE_DATA_DIR/$split.tar.gz"
        echo "Extracting $split..."
        tar -xzf "$BASE_DATA_DIR/$split.tar.gz" -C "$BASE_DATA_DIR"
        rm "$BASE_DATA_DIR/$split.tar.gz"
    else
        echo "$split already exists, skipping."
    fi
done

# --------------------------------------------------
# 2. WHAM! Noise Dataset
# --------------------------------------------------
WHAM_DIR="$BASE_DATA_DIR/wham_noise"
echo "[2/2] Downloading WHAM! noise dataset..."

if [ ! -d "$WHAM_DIR" ]; then
    mkdir -p "$WHAM_DIR"
    
    # 공식 최신 S3 버킷 링크 사용 및 이어받기(-c) 옵션 추가
    WHAM_URL="https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip"
    
    echo "Downloading WHAM! noise zip from official S3 bucket..."
    wget -c -q --show-progress "$WHAM_URL" -O "$BASE_DATA_DIR/wham_noise.zip"
    
    echo "Extracting WHAM! noise..."
    # -j 옵션을 주면 압축 파일 내 불필요한 상위 디렉토리 구조를 무시하고 
    # 바로 정해진 디렉토리에 깔끔하게 풀 수 있습니다.
    unzip -q "$BASE_DATA_DIR/wham_noise.zip" -d "$WHAM_DIR"
    
    rm "$BASE_DATA_DIR/wham_noise.zip"
    echo "WHAM! noise dataset setup complete."
else
    echo "WHAM! noise dataset already exists, skipping."
fi

echo "=================================================="
echo "All done! Please check your config.yaml paths."
echo "=================================================="