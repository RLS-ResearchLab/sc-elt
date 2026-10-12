"""Compute-quality sweep: FID of one trained looped checkpoint at several loop counts.

FID is reported against nfe_backbone = DDIM steps x loops x CFG multiplier
(multiplier 1 without guidance, 2 with it), the true number of backbone passes.
Every loop count uses the same seed, hence the same initial noise and labels.

Example
-------
python src/eval_loop_counts.py --checkpoint checkpoints/<run>/looped_dit_run1_final.pt \
    --loops 1 2 3 4 --num-samples 1000 --steps 50 --output results/loop_sweep_elt.csv
"""

import argparse
import csv
import os

import torch
import torchvision.utils as vutils
from torchmetrics.image.fid import FrechetInceptionDistance

from data import build_dataloader, extract_images
from loading import FixedLoops, load_trained_model
from utils import denormalize, sample_labels


@torch.no_grad()
def generate_batch(net, model, vae, diffusion, scale, n, args, device, y=None):
    shape = (n, model.cfg.grid_h * model.cfg.grid_w, model.in_channels)
    z = diffusion.sample(
        net, shape, y=y, sampler="ddim", num_steps=args.steps,
        guidance_scale=args.guidance, eta=args.eta, device=device,
    )
    return denormalize(vae.decoder(z / scale))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--loops", type=int, nargs="+", default=None,
                   help="Loop counts to evaluate (default: 1 .. trained loop_steps).")
    p.add_argument("--num-samples", type=int, default=1000, help="Generated images per loop count.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=50, help="DDIM steps")
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--real-split", default="test", choices=["test", "train"])
    p.add_argument("--data-root", default=None, help="Override data.root stored in the checkpoint config.")
    p.add_argument("--vae-checkpoint", default=None)
    p.add_argument("--scaling-factor", type=float, default=None)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--output", default="results/loop_sweep.csv")
    p.add_argument("--save-grids", action="store_true",
                   help="Also save a 16-image grid per loop count (same noise) next to the CSV.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    bundle = load_trained_model(
        args.checkpoint, device, use_ema=not args.no_ema,
        vae_checkpoint=args.vae_checkpoint, scaling_factor=args.scaling_factor,
    )
    model, vae, diffusion, cfg = bundle["model"], bundle["vae"], bundle["diffusion"], bundle["cfg"]
    scale = bundle["scaling_factor"]

    if not hasattr(model, "forward_elastic"):
        raise SystemExit("This sweep needs a looped_dit checkpoint.")
    trained_loops = model.loop_cfg.loop_steps
    loops_list = args.loops or list(range(1, trained_loops + 1))
    if max(loops_list) > trained_loops:
        print(f"WARNING: evaluating more loops than trained ({trained_loops}).")
    cfg_mult = 1 if args.guidance == 1.0 else 2

    # Real images are encoded once; reset() keeps them (reset_real_features=False).
    data_cfg = dict(cfg["data"])
    if args.data_root:
        data_cfg["root"] = args.data_root
    loader = build_dataloader(data_cfg, split=args.real_split)
    metric = FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=False).to(device)
    n_real = 0
    for batch in loader:
        images = extract_images(batch).to(device)
        metric.update(denormalize(images), real=True)
        n_real += images.size(0)
    print(f"Real images ({args.real_split} split): {n_real}")

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    for loops in loops_list:
        net = FixedLoops(model, loops)
        torch.manual_seed(args.seed)          # same noise and labels for every loop count
        metric.reset()
        done = 0
        while done < args.num_samples:
            b = min(args.batch_size, args.num_samples - done)
            y = sample_labels(model.effective_num_classes, b, device, mode="random")
            imgs = generate_batch(net, model, vae, diffusion, scale, b, args, device, y=y)
            metric.update(imgs, real=False)
            if args.save_grids and done == 0:
                vutils.save_image(imgs[:16].cpu(), os.path.join(out_dir, f"grid_loops{loops}.png"), nrow=4)
            done += b
        fid = metric.compute().item()
        nfe = args.steps * loops * cfg_mult
        print(f"[loop_count_eval] loops={loops} fid={fid:.4f} nfe_backbone={nfe} nfe_network={args.steps * cfg_mult}")
        rows.append({
            "loops": loops, "ddim_steps": args.steps, "guidance": args.guidance,
            "cfg_multiplier": cfg_mult, "nfe_network": args.steps * cfg_mult, "nfe_backbone": nfe, "fid": round(fid, 3),
            "num_samples": args.num_samples, "seed": args.seed,
            "checkpoint_step": bundle["step"], "ema": not args.no_ema,
        })

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.output}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot([r["nfe_backbone"] for r in rows], [r["fid"] for r in rows], marker="o")
        for r in rows:
            plt.annotate(f"{r['loops']} loops", (r["nfe_backbone"], r["fid"]),
                         textcoords="offset points", xytext=(4, 6), fontsize=8)
        plt.xlabel("nfe_backbone (DDIM steps x loops x CFG multiplier)")
        plt.ylabel("FID")
        plt.title("FID vs backbone passes")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.splitext(args.output)[0] + ".png", dpi=150)
    except ImportError:
        print("matplotlib not installed: skipping the plot.")


if __name__ == "__main__":
    main()