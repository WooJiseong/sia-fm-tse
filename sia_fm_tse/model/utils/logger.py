"""W&B based Logger"""

import wandb


class Logger:
    def __init__(self, **wandb_kwargs):
        try:
            wandb.init(**wandb_kwargs)
            self.enaled = True
        except Exception as e:
            print(f"[Logger] wandb init failed: {e}")
            self.enabled = False

    def log(self, metrics: dict, step: int | None = None):
        if self.enabled:
            wandb.log(metrics, step=step)
        else:
            msg = " | ".join(
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in metrics.items()
            )
            print(f"[step {step}] {msg}")

    def finish(self):
        if self.enabled:
            wandb.finish()
