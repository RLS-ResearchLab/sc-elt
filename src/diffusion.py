import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from math import pi


@dataclass
class DiffusionConfig:
    timestep: int
    schedule: str
    beta_start: float = 1e-4
    beta_end: float = 0.02
    cosine_s: float = 0.008
    learn_sigma: bool = True

    def __post_init__(self):
        assert self.schedule in ("linear", "cosine"), f"Invalid schedule: {self.schedule}"
        assert self.timestep > 0


def build_diffusion(cfg, device=None):
    diffusion_cfg = DiffusionConfig(**cfg["diffusion"])
    return GaussianDiffusion(diffusion_cfg, device=device)


def validate_betas(betas):
    assert betas.ndim == 1
    assert torch.all(betas > 0)
    assert torch.all(betas < 1)


def build_linear_schedule(cfg: DiffusionConfig):
    T = cfg.timestep
    beta_start = cfg.beta_start
    beta_end = cfg.beta_end

    assert beta_start < beta_end, (
        f"beta_end must be greater than beta_start, "
        f"got beta_start:{beta_start}, beta_end:{beta_end}"
    )

    betas = torch.linspace(beta_start, beta_end, T)
    validate_betas(betas)
    return betas


def build_cosine_schedule(cfg: DiffusionConfig):
    T = cfg.timestep
    s = cfg.cosine_s

    steps = torch.arange(T + 1, dtype=torch.float32)
    x = (steps / T + s) / (1 + s)

    alphas_cumprod = torch.cos(x * pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = torch.clamp(betas, 1e-5, 0.999)

    validate_betas(betas)
    return betas


SCHEDULE_BUILDERS = {
    "linear": build_linear_schedule,
    "cosine": build_cosine_schedule,
}


def build_schedule(cfg: DiffusionConfig):
    name = cfg.schedule

    if name not in SCHEDULE_BUILDERS:
        raise ValueError(
            f"Unknown schedule '{name}', "
            f"expected one of {list(SCHEDULE_BUILDERS)}"
        )

    return SCHEDULE_BUILDERS[name](cfg)


class GaussianDiffusion(nn.Module):
    def __init__(self, cfg: DiffusionConfig, device=None):
        super().__init__()
        self.config = cfg

        betas = build_schedule(cfg)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=betas.device), alphas_cumprod[:-1]]
        )

        posterior_variance = (
            betas
            * (1.0 - alphas_cumprod_prev)
            / (1.0 - alphas_cumprod)
        )

        posterior_mean_coef1 = (
            betas
            * torch.sqrt(alphas_cumprod_prev)
            / (1.0 - alphas_cumprod)
        )

        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - alphas_cumprod)
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer(
            "sqrt_alphas_cumprod",
            torch.sqrt(alphas_cumprod),
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_mean_coef1",
            posterior_mean_coef1,
        )
        self.register_buffer(
            "posterior_mean_coef2",
            posterior_mean_coef2,
        )

        log_betas = torch.log(betas)
        posterior_log_variance_clipped = torch.log(
            torch.cat([posterior_variance[1:2], posterior_variance[1:]])
        )

        self.register_buffer("log_betas", log_betas)
        self.register_buffer(
            "posterior_log_variance_clipped",
            posterior_log_variance_clipped,
        )

        if device is not None:
            self.to(device)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ):
        x_shape = x0.shape

        return (
            extract(self.sqrt_alphas_cumprod, t, x_shape) * x0
            + extract(
                self.sqrt_one_minus_alphas_cumprod,
                t,
                x_shape,
            )
            * noise
        )

    def forward_process(self, x0: torch.Tensor):
        B = x0.shape[0]
        t = torch.randint(
            0,
            self.config.timestep,
            (B,),
            device=x0.device,
        )
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        return x_t, t, noise

    def forward(self, x0):
        return self.forward_process(x0)

    def predict_x0(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps_pred: torch.Tensor,
    ):
        x_shape = x_t.shape

        return (
            1 / extract(self.sqrt_alphas_cumprod, t, x_shape)
        ) * (
            x_t
            - extract(
                self.sqrt_one_minus_alphas_cumprod,
                t,
                x_shape,
            )
            * eps_pred
        )

    def compute_learned_variance(
        self,
        v: torch.Tensor,
        t: torch.Tensor,
        x_shape,
    ):
        frac = (v + 1) / 2

        log_var = (
            frac * extract(self.log_betas, t, x_shape)
            + (1 - frac)
            * extract(
                self.posterior_log_variance_clipped,
                t,
                x_shape,
            )
        )

        return torch.exp(log_var), log_var

    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps_pred: torch.Tensor,
        v: torch.Tensor = None,
    ):
        x0_pred = self.predict_x0(x_t, t, eps_pred)
        x_shape = x_t.shape

        mean = (
            extract(self.posterior_mean_coef1, t, x_shape) * x0_pred
            + extract(self.posterior_mean_coef2, t, x_shape) * x_t
        )

        if self.config.learn_sigma and v is not None:
            var, log_var = self.compute_learned_variance(v, t, x_shape)
        else:
            var = extract(self.posterior_variance, t, x_shape)
            log_var = extract(
                self.posterior_log_variance_clipped,
                t,
                x_shape,
            )

        return mean, var, log_var

    @torch.no_grad()
    def ddpm_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        v: torch.Tensor = None,
    ):
        mean, var, _ = self.p_mean_variance(x_t, t, eps, v)
        noise = torch.randn_like(x_t)

        nonzero_mask = (t != 0).float().reshape(
            t.shape[0],
            *((1,) * (x_t.ndim - 1)),
        )

        return mean + nonzero_mask * torch.sqrt(var) * noise

    @torch.no_grad()
    def ddim_step(
        self,
        x_t: torch.Tensor,
        t,
        t_prev,
        eps: torch.Tensor,
        eta: float = 0.0,
    ):
        x_shape = x_t.shape

        alpha_cumprod_t = extract(
            self.alphas_cumprod,
            t,
            x_shape,
        )
        alpha_cumpro_prev = extract(
            self.alphas_cumprod,
            t_prev,
            x_shape,
        )

        x0_pred = self.predict_x0(x_t, t, eps)

        if eta == 0.0:
            sigma_t = torch.zeros_like(alpha_cumprod_t)
        else:
            sigma_t = eta * torch.sqrt(
                (1 - alpha_cumpro_prev)
                / (1 - alpha_cumprod_t)
                * (1 - alpha_cumprod_t / alpha_cumpro_prev)
            )

        direction = torch.sqrt(
            torch.clamp(
                1 - alpha_cumpro_prev - sigma_t**2,
                min=0.0,
            )
        ) * eps

        x_prev = (
            torch.sqrt(alpha_cumpro_prev) * x0_pred
            + direction
        )

        if eta > 0.0:
            noise = torch.randn_like(x_t)
            nonzero_mask = (t != 0).float().reshape(
                t.shape[0],
                *((1,) * (x_t.ndim - 1)),
            )
            x_prev = x_prev + nonzero_mask * sigma_t * noise

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model,
        shape,
        y=None,
        guidance_scale=1.0,
        sampler="ddim",
        num_steps=50,
        eta=0.0,
        device=None,
    ):
        if device is None:
            device = self.betas.device

        x = torch.randn(shape, device=device)

        if sampler == "ddpm":
            timesteps = torch.arange(
                self.config.timestep - 1,
                -1,
                -1,
                device=device,
            )
        elif sampler == "ddim":
            timesteps = torch.linspace(
                self.config.timestep - 1,
                0,
                num_steps,
                device=device,
            ).long()
        else:
            raise ValueError(f"Unknown sampler {sampler}")

        for i, t in enumerate(timesteps):
            t_batch = torch.full(
                (shape[0],),
                t,
                device=device,
                dtype=torch.long,
            )

            if guidance_scale != 1.0:
                eps_cond, v = self._split_output(
                    model(x, t_batch, y)
                )
                eps_uncond, _ = self._split_output(
                    model(x, t_batch, None)
                )
                eps = eps_uncond + guidance_scale * (
                    eps_cond - eps_uncond
                )
            else:
                eps, v = self._split_output(
                    model(x, t_batch, y)
                )

            if sampler == "ddpm":
                x = self.ddpm_step(
                    x,
                    t_batch,
                    eps,
                    v,
                )
            else:
                if i == len(timesteps) - 1:
                    t_prev = torch.zeros_like(t_batch)
                else:
                    t_prev = torch.full(
                        (shape[0],),
                        timesteps[i + 1],
                        device=device,
                        dtype=torch.long,
                    )

                x = self.ddim_step(
                    x,
                    t_batch,
                    t_prev,
                    eps,
                    eta=eta,
                )

        return x

    def _split_output(self, out):
        return out if isinstance(out, tuple) else (out, None)


def extract(
    schedule_buffer: torch.Tensor,
    t: torch.Tensor,
    x_shape,
):
    out = schedule_buffer[t]
    out = out.reshape(
        t.shape[0],
        *((1,) * (len(x_shape) - 1)),
    )
    return out