
CUDA_VISIBLE_DEVICES=1 \
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --master_addr=localhost \
    --master_port=29525 \
    ./train.py \
    -conf config/train.yaml


