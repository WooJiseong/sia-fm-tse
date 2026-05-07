#=====[ 의존성 충돌 해결용 Dockerfile ]=====#
ARG CUDA_VERSION=12.1.0
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04

# ESPNet에서 3.12를 지원함!
# https://github.com/espnet/espnet
ARG PYTHON_VERSION=3.12

# tzdata가 인터랙티브 띄워서 강제
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 기본 패키지 및 python 설치
#   - deadsnakes PPA에서 <=3.11 버전 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    && rm -rf /var/lib/apt/lists/*


# uv 설치
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Python 버전 고정
RUN uv python pin ${PYTHON_VERSION}

WORKDIR /workspace

# Python 의존성 설치
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen 2>/dev/null || uv sync    # Lockfile 우선 -> 일반

CMD ["bash"]
