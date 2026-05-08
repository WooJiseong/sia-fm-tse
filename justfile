# 도움말
default:
    @just --list

# 개발 환경 셋업
setup:
    uv sync --frozen && \
    uv pip install . -e --no-deps
