build:
    docker build --build-arg CUDA_VERSION=${CUDA_VERSION:-12.1.0} -t sia-fm-tse .

dev:
    docker run --gpus all -it -v .:/workspace sia-fm-tse

train:
    docker run --gpus all -v .:/workspace sia-fm-tse uv run python -m src.train
