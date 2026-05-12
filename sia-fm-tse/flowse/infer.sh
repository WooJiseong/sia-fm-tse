
config_path=config/train.yaml
CUDA_VISIBLE_DEVICES=3 \
python \
    ./infer.py \
    -conf ${config_path}

