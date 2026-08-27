import argparse
import torch
import yaml

from data import build_latent_cache
from utils import build_vae_from_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Loading VAE: {args.vae_checkpoint}")

    vae = build_vae_from_checkpoint(
        args.vae_checkpoint,
        device=device,
        freeze=True,
    )

    build_latent_cache(
        encoder=vae.encoder,
        cfg=cfg["data"],
        split=args.split,
        device=device,
        chunk_size=9984,
    )

    print("Latent cache created successfully.")


if __name__ == "__main__":
    main()
