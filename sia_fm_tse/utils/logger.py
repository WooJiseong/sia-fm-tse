"""W&B based logging handler."""

import logging

import wandb
from wandb.sdk import AlertLevel


class WandbHandler(logging.Handler):
    """A logging handler that routes log records to Weights & Biases.

    - WARNING and above  → wandb.alert()  (one-shot events: non-finite loss, load failures, etc.)
    - INFO and below     → wandb.run.notes / console only; not forwarded to avoid noise
    """

    # Levels that deserve a W&B alert rather than a metric log.
    _ALERT_LEVELS = frozenset({logging.WARNING, logging.ERROR, logging.CRITICAL})

    _WANDB_ALERT_LEVEL = {
        logging.WARNING: AlertLevel.WARN,
        logging.ERROR: AlertLevel.ERROR,
        logging.CRITICAL: AlertLevel.ERROR,
    }

    def emit(self, record: logging.LogRecord) -> None:
        if wandb.run is None:
            return

        msg = self.format(record)

        if record.levelno in self._ALERT_LEVELS:
            wandb.run.alert(
                title=f"[{record.module}] {record.levelname}",
                text=msg,
                level=self._WANDB_ALERT_LEVEL[record.levelno],
            )
        else:
            # INFO-level logs are printed to the console by the stream handler;
            # forwarding every progress line to W&B would pollute the dashboard.
            pass
