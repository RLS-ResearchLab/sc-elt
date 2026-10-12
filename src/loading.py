"""Generate a grid of images from a trained DiT / Looped-DiT / ELT checkpoint.

Examples
--------
python src/generate.py --checkpoint checkpoints/<run>/looped_dit_run1_final.pt
python src/generate.py --checkpoint <ckpt> --loops 3 --steps 50 --num-images 16 --out samples/loops3.png
"""

import argparse
import os

import torch
import torchvision.utils as vutils

from loading import FixedLoops, load_trained_model
from utils import denormalize, sample_labels


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="samples/generated.png")
    p.add_argument("--num-images", type=int, default=16)
    p.add_argument("--nrow", type=int, default=4)
    p.add_argument("--steps", type=int, default=50, help="DDIM steps")
    p.add_argument("--guidance", type=float, default=1.0, help="CFG scale (1.0 = off)")
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--loops", type=int, default=None,
                   help="Backbone loops at inference (LoopedDiT only). Default: the trained loop_steps.")
    p.add_argument("--labels", choices=["random", "cycle", "none"], default="random",
                   help="Class labels fed to the model (training-time FID uses random).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vae-checkpoint", default=None, help="Override the VAE path stored in the checkpoint config.")
    p.add_argument("--scaling-factor", type=float, default=None)
    p.add_argument("--no-ema", action="store_true", help="Use raw weights instead of EMA weights.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    bundle = load_trained_model(
        args.checkpoint, device,
        use_ema=not args.no_ema,
        vae_checkpoint=args.vae_checkpoint,
        scaling_factor=args.scaling_factor,
    )
    model, vae, diffusion = bundle["model"], bundle["vae"], bundle["diffusion"]

    net = FixedLoops(model, args.loops) if args.loops is not None else model
    loops = args.loops if args.loops is not None else getattr(getattr(model, "loop_cfg", None), "loop_steps", 1)
    cfg_mult = 1 if args.guidance == 1.0 else 2
    print(f"checkpoint step: {bundle['step']} | loops: {loops} | "
          f"nfe_backbone = {args.steps} x {loops} x {cfg_mult} = {args.steps * loops * cfg_mult}")

    torch.manual_seed(args.seed)
    n = args.num_images
    shape = (n, model.cfg.grid_h * model.cfg.grid_w, model.in_channels)
    y = None if args.labels == "none" else sample_labels(
        model.effective_num_classes, n, device, mode=args.labels)

    with torch.no_grad():
        z = diffusion.sample(
            net, shape, y=y, sampler="ddim", num_steps=args.steps,
            guidance_scale=args.guidance, eta=args.eta, device=device,
        )
        images = denormalize(vae.decoder(z / bundle["scaling_factor"]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    vutils.save_image(images.cpu(), args.out, nrow=args.nrow)
    print(f"Saved {n} images to {args.out}")


if __name__ == "__main__":
    main()