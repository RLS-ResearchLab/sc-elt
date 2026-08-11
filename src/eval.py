import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from dataclasses import dataclass

from utils import sample_labels, denormalize
from data import extract_images


# ============================================================
# Evaluation Configs
# ============================================================

@dataclass
class ReconstructionEvalConfig:
    enabled: bool = True
    num_images: int = 8


@dataclass
class FIDEvalConfig:
    enabled: bool = True
    fid_num_samples: int = 1000
    sample_batch_size: int = 64
    ddim_steps: int = 50
    guidance_scale: float = 1.0
    eta: float = 0.0


@dataclass
class VAEEvalConfig:
    reconstruction: ReconstructionEvalConfig


@dataclass
class DiTEvalConfig:
    fid: FIDEvalConfig


@dataclass
class ELTEvalConfig:
    """Placeholder for future ELT evaluation configuration."""
    pass


# ============================================================
# Base Evaluator
# ============================================================

class Evaluator:
    """Base class for all evaluators."""

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

    @torch.no_grad()
    def evaluate(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement evaluate()."
        )


# ============================================================
# Reconstruction Evaluator
# ============================================================

class ReconstructionEvaluator(Evaluator):

    def __init__(
        self,
        cfg,
        model,
        dataloader,
        device,
    ):
        super().__init__(cfg, device)
        self.model = model
        self.dataloader = dataloader

    @torch.no_grad()
    def evaluate(self):
        was_training = self.model.training
        self.model.eval()

        total_loss = 0.0
        count = 0

        originals = []
        reconstructions = []

        try:
            for batch in self.dataloader:
                images = extract_images(batch)
                images = images.to(self.device, non_blocking=True)

                output = self.model(images)
                recon = output["recon"]

                mse = torch.mean((images - recon) ** 2)

                total_loss += mse.item()
                count += 1

                # Store images for visualization.
                if len(originals) < self.cfg.num_images:
                    remaining = self.cfg.num_images - len(originals)

                    originals.append(images[:remaining].cpu())
                    reconstructions.append(recon[:remaining].cpu())

        finally:
            self.model.train(was_training)

        originals = torch.cat(
            originals,
            dim=0,
        )[:self.cfg.num_images]

        reconstructions = torch.cat(
            reconstructions,
            dim=0,
        )[:self.cfg.num_images]

        return {
            "metrics": {
                "reconstruction_mse": total_loss / count,
            },
            "images": {
                "originals": originals,
                "reconstructions": reconstructions,
            },
        }


# ============================================================
# FID Evaluator
# ============================================================

class FIDEvaluator(Evaluator):

    def __init__(
        self,
        cfg,
        fid_metric,
        real_loader,
        model,
        vae,
        diffusion,
        scaling_factor,
        device,
    ):
        super().__init__(cfg, device)

        self.metric = fid_metric
        self.real_loader = real_loader
        self.model = model
        self.vae = vae
        self.scaling_factor = scaling_factor
        self.diffusion = diffusion

    @torch.no_grad()
    def evaluate(self):
        was_training = self.model.training
        self.model.eval()

        try:
            self.metric.reset()

            # Real images.
            for batch in self.real_loader:
                images = extract_images(batch)
                images = images.to(self.device)

                # [-1, 1] -> [0, 1].
                images = denormalize(images)
                self.metric.update(images, real=True)

            # Generated images.
            generated = 0

            while generated < self.cfg.fid_num_samples:
                batch_size = min(
                    self.cfg.sample_batch_size,
                    self.cfg.fid_num_samples - generated,
                )

                y = sample_labels(
                    self.model.effective_num_classes,
                    batch_size,
                    self.device,
                    mode="random",
                )

                z = self.diffusion.sample(
                    self.model,
                    shape=(
                        batch_size,
                        self.model.cfg.grid_h * self.model.cfg.grid_w,
                        self.model.in_channels,
                    ),
                    y=y,
                    sampler="ddim",
                    num_steps=self.cfg.ddim_steps,
                    guidance_scale=self.cfg.guidance_scale,
                    eta=self.cfg.eta,
                    device=self.device,
                )

                images = self.vae.decoder(z / self.scaling_factor)
                images = denormalize(images)

                self.metric.update(images, real=False)
                generated += batch_size

            fid = self.metric.compute().item()

        finally:
            self.model.train(was_training)

        return {
            "metrics": {
                "fid": fid,
            },
        }


# ============================================================
# VAE Evaluator Builder
# ============================================================

def build_vae_evaluators(
    cfg,
    model,
    loaders,
    device,
):
    evaluators = {}

    if cfg.reconstruction.enabled:
        evaluators["reconstruction"] = ReconstructionEvaluator(
            cfg=cfg.reconstruction,
            model=model,
            dataloader=loaders["test"],
            device=device,
        )

    return evaluators


# ============================================================
# DiT Evaluator Builder
# ============================================================

def build_dit_evaluators(
    cfg,
    model,
    vae,
    diffusion,
    scaling_factor,
    loaders,
    fid_metric,
    device,
):
    evaluators = {}

    if cfg.fid.enabled:
        evaluators["fid"] = FIDEvaluator(
            cfg=cfg.fid,
            fid_metric=fid_metric,
            real_loader=loaders["test"],
            model=model,
            vae=vae,
            diffusion=diffusion,
            scaling_factor=scaling_factor,
            device=device,
        )

    return evaluators


# ============================================================
# ELT Evaluator Builder
# ============================================================

def build_elt_evaluators(
    cfg,
    model,
    vae,
    diffusion,
    loaders,
    device,
    fid_metric=None,
):
    raise NotImplementedError(
        "ELT evaluator builder not implemented."
    )


# ============================================================
# Evaluator Factory
# ============================================================

def build_evaluators(cfg, model, loaders, device, **kwargs):
    model_name = cfg["model"]["name"]

    if model_name == "vae":
        eval_cfg = VAEEvalConfig(
            reconstruction=ReconstructionEvalConfig(
                **cfg["eval"]["reconstruction"],
            ),
        )

        return build_vae_evaluators(
            cfg=eval_cfg,
            model=model,
            loaders=loaders,
            device=device,
        )

    if model_name == "dit":
        eval_cfg = DiTEvalConfig(
            fid=FIDEvalConfig(
                **cfg["eval"]["fid"],
            ),
        )

        return build_dit_evaluators(
            cfg=eval_cfg,
            model=model,
            loaders=loaders,
            fid_metric=kwargs.get("fid_metric"),
            vae=kwargs["vae"],
            diffusion=kwargs["diffusion"],
            scaling_factor=kwargs["scaling_factor"],
            device=device,
        )

    if model_name == "elt":
        raise NotImplementedError(
            "ELT evaluator builder not implemented."
        )

    raise ValueError(f"Unknown model '{model_name}'.")


# ============================================================
# Utilities
# ============================================================

def visualize_reconstruction(
    originals,
    reconstructions,
    save_path=None,
    show=False,
):
    comparison = torch.cat(
        [originals, reconstructions],
        dim=0,
    )

    grid = vutils.make_grid(
        comparison,
        nrow=len(originals),
        normalize=True,
        value_range=(-1, 1),
    )

    if save_path is not None:
        vutils.save_image(grid, save_path)

    if show:
        plt.figure(figsize=(12, 4))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis("off")
        plt.show()

    return grid


def build_fid_metric(cfg, device):
    from torchmetrics.image.fid import FrechetInceptionDistance

    metric = FrechetInceptionDistance(
        feature=2048,
        normalize=True,
    )

    metric.to(device)

    return metric