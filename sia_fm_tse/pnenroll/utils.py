import yaml
import importlib
import json
import os
from typing import Any, Dict, Optional
import logging
import wandb

def get_config(yaml_config_filename):
    base_dir = "configs/experiment_setting" # 기본 파일명으로 세팅

    if not yaml_config_filename.startswith("configs/"):
        yaml_config_filename = os.path.join(base_dir, yaml_config_filename)

    with open(yaml_config_filename) as f:
        config_dict = yaml.safe_load(f)

    return config_dict

def import_attr(import_path):
    module, attr = import_path.rsplit('.', 1)
    return getattr(importlib.import_module(module), attr)

class Params():
    """ NOTE: Code from the LookOnceToHear paper
    Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

class WandbLogger:

    def __init__(
        self,
        name: str,
        wandb_config: dict, # [get_config로 파싱 필요] Wandb_setting.yaml dict 입력
        config_dict: Optional[dict] = None, # 실험 세팅 하이퍼파라미터가 존재하는 yaml파일의 dict 입력
        format_str: str = "%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
        date_format: str = "%Y-%m-%d %H:%M:%S",
        log_to_file: bool = False, # 로그를 파일로 남길건지를 확인
    ):
        self.name = name

        # 1. W&B 설정 파싱
        w_settings = wandb_config.get("wandb_settings", {})
        self.project = w_settings.get("project", "default-project")
        self.entity = w_settings.get("entity", None)
        self.notes = w_settings.get("notes", None)
        self.tags = w_settings.get("tags", None)
        self.mode = w_settings.get("mode", "online")
        self.config_dict = config_dict

        # 2. 로컬 파이썬 표준 로거 설정
        self.local_logger = logging.getLogger(name)
        self.local_logger.setLevel(logging.INFO)

        if not self.local_logger.handlers:
            formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.local_logger.addHandler(stream_handler)

            if log_to_file:
                file_handler = logging.FileHandler(f"{name}.log")
                file_handler.setFormatter(formatter)
                self.local_logger.addHandler(file_handler)

        self.run = None

    def __enter__(self):
        """with 문에 진입할 때 실행되는 로직 (wandb.init 호출)"""
        if self.run is None:
            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.name,
                notes=self.notes,
                tags=self.tags,
                mode=self.mode,
                config=self.config_dict,
            )
            self.log_info(f"W&B Run '{self.name}' started via Context Manager.")
        return self  # with logger_instance as [] << 에 올 객체로 자기 자신(logger)을 반환

    def __exit__(self, exc_type, exc_val, exc_tb):
        """with 블록 탈출 시 실행 로직 (예외 로깅 / wandb.finish)"""
        if exc_type is not None:
            # 블록 내부 에러 발생 시 로깅 처리
            self.log_error(
                f"Run interrupted by exception: {exc_type.__name__}: {exc_val}"
            )

        # W&B 세션을 안전하게 종료 및 반납 (누수방지) (Failed 상태 등이 W&B에 기록됨)
        self.log_info("Finishing W&B run execution via Context Manager...")
        wandb.finish()
        
        return False

    # ----------------------------------------------------------------

    def log_info(self, message: str):
        self.local_logger.info(message)

    def log_error(self, message: str):
        self.local_logger.error(message)

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)