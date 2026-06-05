"""CFM Decoder Implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchdiffeq import odeint

from .transformer import DiT


class CFM(nn.Module):
    def __init__(
        self,
        *,
        transformer: DiT,
        odeint_method: str = "euler",
        cond_drop_prob: float = 0.0,
    ):
        """
        Continuous Flow Matching (CFM) wrapper around DiT.

        Trains the transformer to predict the flow field (x1 - x0),
        and samples via ODE integration at inference.

        Args:
            transformer: DiT backbone
            odeint_conf: ODE solver configuration
            cond_drop_prob: condition drop probability for CFG training
        """
        super().__init__()

        # classifier-free guidance
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer

        self.odeint_method = odeint_method

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def forward(
        self,
        m: torch.Tensor,
        c: torch.Tensor,
        *,
        steps=32,
        cfg_strength=1.0,
        vocoder: nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample clean audio via ODE integration.

        Args:
            m: [B, T, mel_dim] audio mixture
            c: [B, T, D] encoder condition
            steps: number of ODE steps (=NFE)
            cfg_strength: CFG guidance strength
            vocoder: optional vocoder to convert mel to waveform

        Returns:
            out: [B, T, mel_dim] or [B, nw] if vocoder is provided
            trajectory: [steps+1, B, T, mel_dim] ODE trajectory
        """

        def fn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            t = t.expand(x.shape[0])

            pred = self.transformer(
                x=x,
                m=m,
                c=c,
                t=t,
                mask=None,
            )

            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x,
                m=m,
                c=torch.zeros_like(c),
                t=t,
                mask=None,
            )

            return pred + (pred - null_pred) * cfg_strength

        x0 = torch.randn_like(m)

        t = torch.linspace(0, 1, steps + 1, device=self.device, dtype=c.dtype)
        trajectory: torch.Tensor = odeint(fn, x0, t, method=self.odeint_method)  # type: ignore
        out = trajectory[-1]

        if vocoder is not None:
            out = rearrange(out, "b n d -> b d n")
            out: torch.Tensor = vocoder(out)

        return out, trajectory

    def loss(
        self,
        m: torch.Tensor,
        c: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CFM training loss.

        Args:
            m:  [B, T, mel_dim] audio mixture
            c:  [B, T, D] encoder condition
            x1: [B, T, mel_dim] clean target audio

        Returns:
            loss: scalar MSE loss between predicted and target flow
        """
        x0 = torch.randn_like(x1)
        t = torch.rand(x0.shape[0], dtype=x0.dtype, device=self.device)
        t_expand = rearrange(t, "b -> b 1 1")

        # sample x_t
        φ = (1 - t_expand) * x0 + t_expand * x1
        flow = x1 - x0

        drop_cond_mask = (
            torch.rand(
                x1.shape[0],
                device=self.device,
            )
            < self.cond_drop_prob
        )
        drop_cond_mask = rearrange(drop_cond_mask, "b -> b 1 1 1")
        c = torch.where(drop_cond_mask, torch.zeros_like(c), c)

        pred = self.transformer(x=φ, m=m, c=c, t=t)

        loss = F.mse_loss(pred, flow, reduction="none")

        return loss.mean()
