# 실험을 위한 환경 관리

UBAI 환경은 기본적으로 `conda` 지향적입니다. `uv`를 바탕으로 프로세스를 설계하는 것은 조금 다른 영역의 문제가 될 수 있습니다.

---

## Docker

UBAI는 `enroot`를 사용하여 계산 노드의 컨테이너를 결정합니다. 기본적으로는 그냥 [Docker Hub](https://hub.docker.com/)의 ubuntu-python 이미지를 사용합니다만, [`Dockerfile`](../Dockerfile)을 통해 특정 패키지나 설정을 추가할 수 있습니다.

우리의 Dockerfile이 하는 역할은 다음과 같습니다.

1. 컨테이너에 자체적으로 CUDA 12.1.0을 설치하여 버전을 강제합니다.
2. Python 3.12와 의존성 패키지들을 uv로 설치하고 버전을 고정합니다.

> [!NOTE]
> 개발 환경도 아닌데 pip와 requirements.txt를 사용하지 않고 uv를 사용하는지 의문이 드실 수도 있습니다.
> 
> 재현성 측면에서 [uv.lock](../uv.lock)은 훨씬 더 강력한 제약을 제공합니다. 우리가 uv를 사용하는 것은 pyproject.toml과 requirments.txt를 중복으로 생성하지 않으면서도 강한 제약을 발휘하기 위함입니다.

> [!WARNING]
> Docker 이미지는 **오로지 실험**을 위해 개발되었습니다. 개발 프로세스에서는 컨테이너를 사용할 필요가 없으며 애초에 우리 서버 내에서 docker 자체가 허용되지 않을 수도 있습니다. 따라서 헤드 노드에 접속했을 때에는 `uv sync`로 필요한 패키지를 설치한 후 실행해야 합니다.

---

## Github Actions

앞서 정의한 Docker 이미지를 사용하기 위해서 우리 ghcr.io를 참조합니다. Dockerfile이나 의존성 패키지가 수정되면 push 시점에 [Github Actions](../github/workflows/docker.yaml)에 의해 자동으로 [빌드](ghcr.io/woojiseong/sia-fm-tse:latest)됩니다.

---

## Experiment Process

실제로 실험을 할 때에는 환경 설정을 우선적으로 해주는 것이 중요합니다.

1. `srun`으로 특정 계산 노드에 접속합니다.
2. `enroot import docker://ghcr.io/woojiseong/sia-fm-tse:latest`로 우리의 Docker 이미지를 불러옵니다.
3. `enroot create -n sia-fm-tse sia-fm-tse+latest.sqsh`로 컨테이너를 생성합니다.
4. `enroot start --root --rw --mount .:/workspace sia-fm-tse /bin/bash`로 컨테이너에 진입합니다.
   이 때 마운트 위치 `/workspace`는 Dockerfile이 정의한 프로젝트의 위치입니다.
5. `uv pip install -e . --no-deps`로 현재 프로젝트를 불러옵니다.
6. **실험이 모두 끝난 후**에는 `exit`으로 컨테이너를 종료할 수 있습니다.

따라서 위 프로세스는 `sbatch`로 실험이 전송되기 전에 불러와져야 하며, 간단히 `just setup {node}`로 실행할 수 있습니다.

> [!NOTE]
> 이렇게 하면 우리가 실험을 실행하려 시도했다가 문제가 발생하더라도 코드를 수정하면 바로 해당 계산 노드에 반영됩니다!

> [!TIP]
> [tmux](https://github.com/tmux/tmux)나 [zellij](https://github.com/zellij-org/zellij)를 사용하여 컨테이너를 백그라운드에서 실행할 것을 **강력히 권장**합니다.

---

## Setup at Head Node

한 편,

1. `just`를 사용하기 위해서
2. private repo에 공개한 우리의 Docker 이미지를 사용하기 위해서

헤드 노드에도 사전작업이 필요합니다. github cli와 justfile을 설치하세요.

```sh
conda install -c conda-forge gh just
```

이후 github에 로그인하여 `credential`을 헤드 노드에 저장해야 합니다!

```sh
gh auth login
```

이거까지 해주시면, `just setup`이 정상적으로 동작합니다.
