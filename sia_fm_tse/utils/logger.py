"""W&B based logging handler."""

import logging

import wandb
from wandb.sdk import AlertLevel


class WandbHandler(logging.Handler):
    """A logging handler that routes log records to Weights & Biases.
    Runs in offline mode by default. To upload later: `wandb sync wandb/`
    - dict msg           -> wandb.log()  (metrics)
    - WARNING and above  -> wandb.alert()  (non-finite loss, load failures, etc.)
    - INFO and below     -> console only via stream handler
    """

    _ALERT_LEVELS = frozenset({logging.WARNING, logging.ERROR, logging.CRITICAL})

    _WANDB_ALERT_LEVEL = {
        logging.WARNING: AlertLevel.WARN,
        logging.ERROR: AlertLevel.ERROR,
        logging.CRITICAL: AlertLevel.ERROR,
    }

    def __init__(self, project: str = "sia-fm-tse", **wandb_kwargs):
        super().__init__()
        wandb.init(project=project, mode="offline", **wandb_kwargs)

    def emit(self, record: logging.LogRecord) -> None:
        if wandb.run is None:
            return

        if isinstance(record.msg, dict):
            wandb.log(record.msg)
            return

        if record.levelno in self._ALERT_LEVELS:
            wandb.run.alert(
                title=f"[{record.module}] {record.levelname}",
                text=self.format(record),
                level=self._WANDB_ALERT_LEVEL[record.levelno],
            )
