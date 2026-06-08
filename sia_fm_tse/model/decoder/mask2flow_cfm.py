"""Flow matcher that refines a coarse target estimate to the clean source."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchdiffeq import odeint

from .mask2flow_transformer import Mask2FlowDiT


class Mask2FlowCFM(nn.Module):
    def __init__(
        self,
        *,
        transformer: Mask2FlowDiT,
        odeint_method: str = "euler",
        cond_drop_prob: float = 0.0,
    ):
        super().__init__()
        self.transformer = transformer
        self.odeint_method = odeint_method
        self.cond_drop_prob = cond_drop_prob

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def forward(
        self,
        start: torch.Tensor,
        mixture: torch.Tensor,
        c: torch.Tensor,
        *,
        steps: int = 1,
        cfg_strength: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate from coarse estimate S0 to target source S."""

        def fn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            t = t.expand(x.shape[0])
            pred = self.transformer(x=x, mixture=mixture, c=c, t=t, mask=None)

            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x,
                mixture=mixture,
                c=torch.zeros_like(c),
                t=t,
                mask=None,
            )
            return pred + (pred - null_pred) * cfg_strength

        t = torch.linspace(0, 1, steps + 1, device=self.device, dtype=c.dtype)
        trajectory: torch.Tensor = odeint(fn, start, t, method=self.odeint_method)  # type: ignore
        return trajectory[-1], trajectory

    def loss(
        self,
        start: torch.Tensor,
        mixture: torch.Tensor,
        c: torch.Tensor,
        target: torch.Tensor,
        *,
        endpoint_l1_weight: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return total loss and detached scalar components.

        start:   [B, T, D] coarse target estimate S0
        mixture: [B, T, D] mixture STFT Y
        target:  [B, T, D] clean target STFT S
        """
        x0 = start
        x1 = target
        t = torch.rand(x0.shape[0], dtype=x0.dtype, device=self.device)
        t_expand = rearrange(t, "b -> b 1 1")

        xt = (1 - t_expand) * x0 + t_expand * x1
        flow = x1 - x0

        drop_cond_mask = torch.rand(x1.shape[0], device=self.device) < self.cond_drop_prob
        drop_cond_mask = rearrange(drop_cond_mask, "b -> b 1 1 1")
        c = torch.where(drop_cond_mask, torch.zeros_like(c), c)

        pred = self.transformer(x=xt, mixture=mixture, c=c, t=t)
        flow_loss = F.mse_loss(pred, flow, reduction="mean")

        # For a straight path, x1 = xt + (1 - t) * v.
        endpoint = xt + (1 - t_expand) * pred
        endpoint_l1 = F.l1_loss(endpoint, x1)

        total = flow_loss + endpoint_l1_weight * endpoint_l1
        metrics = {
            "flow_loss": flow_loss.detach(),
            "endpoint_l1": endpoint_l1.detach(),
            "total_loss": total.detach(),
        }
        return total, metrics
