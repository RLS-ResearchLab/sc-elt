from dataclasses import dataclass

import torch

from data import extract_images


# ============================================================
# Sampling Configs
# ============================================================

@dataclass
class ReconstructionSamplingConfig:
    """Reconstruct images from an input dataloader."""
    enabled: bool = True
    num_images: int = 8
    save_dir: str = "samples"


@dataclass
class RandomSamplingConfig:
    """Decode randomly sampled latent vectors."""
    enabled: bool = False
    num_images: int = 8
    save_dir: str = "samples"


@dataclass
class DiffusionSamplingConfig:
    """Generate images from a diffusion model."""
    enabled: bool = True
    num_images: int = 64
    batch_size: int = 64
    sampler: str = "ddim"
    num_steps: int = 50
    guidance_scale: float = 1.0
    eta: float = 0.0
    save_dir: str = "samples"


@dataclass
class VAESamplingConfig:
    reconstruction: ReconstructionSamplingConfig
    random: RandomSamplingConfig


@dataclass
class DiTSamplingConfig:
    diffusion: DiffusionSamplingConfig


@dataclass
class ELTSamplingConfig:
    """Placeholder for future ELT sampling configuration."""
    pass


# ============================================================
# Base Sampler
# ============================================================

class Sampler:
    """Base class for all samplers."""

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

    @torch.no_grad()
    def generate(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement generate()."
        )


# ============================================================
# VAE Sampler
# ============================================================

class VAESampler(Sampler):

    def __init__(self, cfg, model, dataloader, device):
        super().__init__(cfg, device)
        self.model = model
        self.dataloader = dataloader

    def generate(self):
        outputs = {}

        if self.cfg.reconstruction.enabled:
            outputs["reconstruction"] = self._generate_reconstruction()

        if self.cfg.random.enabled:
            outputs["random"] = self._generate_random()

        return outputs

    def _generate_reconstruction(self):
        was_training = self.model.training
        self.model.eval()

        originals = []
        reconstructions = []
        num_images = 0
        target = self.cfg.reconstruction.num_images

        try:
            for batch in self.dataloader:
                images = extract_images(batch)
                images = images.to(self.device, non_blocking=True)

                output = self.model(images)
                recon = output["recon"]

                remaining = target - num_images

                originals.append(images[:remaining].cpu())
                reconstructions.append(recon[:remaining].cpu())

                num_images += min(images.size(0), remaining)

                if num_images >= target:
                    break

        finally:
            self.model.train(was_training)

        originals = torch.cat(originals, dim=0)[:target]
        reconstructions = torch.cat(reconstructions, dim=0)[:target]

        return {
            "images": {
                "originals": originals,
                "reconstructions": reconstructions,
            }
        }

    @torch.no_grad()
    def _generate_random(self):
        was_training = self.model.training
        self.model.eval()

        try:
            vae_cfg = self.model.encoder.config

            z = torch.randn(
                self.cfg.random.num_images,
                vae_cfg.grid_h * vae_cfg.grid_w,
                vae_cfg.latent_dim,
                device=self.device,
            )

            images = self.model.decoder(z)

        finally:
            self.model.train(was_training)

        return {"images": images.cpu()}


# ============================================================
# DiT Sampler
# ============================================================

class DiTSampler(Sampler):

    def __init__(
        self,
        cfg,
        model,
        vae,
        diffusion,
        scaling_factor,
        device,
        shape,
    ):
        super().__init__(cfg, device)
        self.model = model
        self.vae = vae
        self.diffusion = diffusion
        self.shape = shape
        self.scaling_factor = scaling_factor

    @torch.no_grad()
    def generate(self):
        outputs = {}

        if self.cfg.diffusion.enabled:
            outputs["diffusion"] = self._generate_diffusion()

        return outputs

    @torch.no_grad()
    def _generate_diffusion(self):
        self.model.eval()
        self.vae.eval()

        num_images = self.cfg.diffusion.num_images
        batch_size = self.cfg.diffusion.batch_size

        all_latents = []
        all_images = []
        generated = 0

        while generated < num_images:
            this_batch = min(batch_size, num_images - generated)
            shape = (this_batch, *self.shape)

            latents = self.diffusion.sample(
                self.model,
                shape,
                sampler=self.cfg.diffusion.sampler,
                num_steps=self.cfg.diffusion.num_steps,
                guidance_scale=self.cfg.diffusion.guidance_scale,
                eta=self.cfg.diffusion.eta,
                device=self.device,
            )

            images = self.vae.decoder(latents / self.scaling_factor)

            all_latents.append(latents.cpu())
            all_images.append(images.cpu())
            generated += this_batch

        return {
            "latents": torch.cat(all_latents, dim=0),
            "images": torch.cat(all_images, dim=0),
        }


# ============================================================
# Sampler Factory
# ============================================================

def build_sampler(
    cfg,
    model,
    device,
    vae=None,
    diffusion=None,
    scaling_factor=None,
    dataloader=None,
):
    name = cfg["model"]["name"]

    if name == "vae":
        sampling_cfg = VAESamplingConfig(
            reconstruction=ReconstructionSamplingConfig(
                **cfg["sampling"]["reconstruction"]
            ),
            random=RandomSamplingConfig(
                **cfg["sampling"]["random"]
            ),
        )

        return VAESampler(
            cfg=sampling_cfg,
            model=model,
            dataloader=dataloader,
            device=device,
        )

    if name == "dit":
        sampling_cfg = DiTSamplingConfig(
            diffusion=DiffusionSamplingConfig(
                **cfg["sampling"]["diffusion"]
            )
        )

        shape = (
            cfg["model"]["dit"]["grid_h"] * cfg["model"]["dit"]["grid_w"],
            cfg["model"]["dit"]["latent_dim"],
        )

        return DiTSampler(
            cfg=sampling_cfg,
            model=model,
            vae=vae,
            diffusion=diffusion,
            scaling_factor=scaling_factor,
            device=device,
            shape=shape,
        )

    raise ValueError(f"Unknown sampler for model {name}")


# ============================================================
# Utilities
# ============================================================

def tokens_to_latent(x, grid_h, grid_w):
    B, N, C = x.shape

    assert N == grid_h * grid_w

    x = x.reshape(B, grid_h, grid_w, C)
    x = x.permute(0, 3, 1, 2)

    return x