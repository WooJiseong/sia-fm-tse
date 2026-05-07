IMAGE := "ghcr.io/woojiseong/sia-fm-tse:latest"
CONTAINER := "sia-fm-tse"

# 도움말
default:
    @just --list

# 이미지 빌드 (Dockerfile 수정할 때만 사용)
build:
    docker build --build-arg CUDA_VERSION=${CUDA_VERSION:-12.1.0} -t sia-fm-tse .

# 계산 노드에서 실험 환경 셋업: e.g. `just setup gpu5`
setup node:
    srun --pty --nodelist={{ node }} /bin/bash -c \
        "enroot import docker://{{ IMAGE }} && \
         enroot create -n {{ CONTAINER }} {{ CONTAINER }}+latest.sqsh && \
         enroot start --root --rw --mount .:/workspace {{ CONTAINER }} \
             /bin/bash -c 'cd /workspace && uv pip install -e . --no-deps'"
