import copy
import math
import os
import glob
import random

import numpy as np
import torch
import torch.nn.functional as F
import wandb
import yaml

from model import build_model


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(default_cfg, stage_cfg):
    merged = copy.deepcopy(default_cfg)

    for key, value in stage_cfg.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def load_config(default_path, stage_path):
    default_cfg = load_yaml(default_path)
    stage_cfg = load_yaml(stage_path)
    return merge_configs(default_cfg, stage_cfg)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_next_run_number(base_dir, run_name):
    if not os.path.exists(base_dir):
        return 1

    prefix = f"{run_name}_run"
    existing = [d for d in os.listdir(base_dir) if d.startswith(prefix)]

    run_nums = []
    for d in existing:
        suffix = d[len(prefix):]
        if suffix.isdigit():
            run_nums.append(int(suffix))

    return max(run_nums, default=0) + 1


def build_experiment_name(cfg):
    model = cfg["model"]["name"]
    dataset = cfg["data"]["name"]
    size = cfg["data"]["image_size"]
    return f"{model}_{dataset}_{size}"


def setup_environment(cfg):
    train_cfg = cfg["train"]

    torch.backends.cudnn.benchmark = train_cfg.get("benchmark", True)
    torch.backends.cudnn.deterministic = train_cfg.get("deterministic", False)

    allow_tf32 = train_cfg.get("allow_tf32", True)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

    base_dir = train_cfg.get("checkpoints_dir", "checkpoints")
    experiment_name = build_experiment_name(cfg)

    run_number = train_cfg.get("run_number")
    if run_number is None:
        run_number = get_next_run_number(base_dir, experiment_name)

    checkpoint_dir = os.path.join(
        base_dir,
        f"{experiment_name}_run{run_number}",
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    cfg["run_number"] = run_number

    return checkpoint_dir


def build_accelerator(cfg):
    from accelerate import Accelerator

    train_cfg = cfg["train"]

    return Accelerator(
        mixed_precision=train_cfg.get("mixed_precision", "bf16"),
        gradient_accumulation_steps=train_cfg.get(
            "gradient_accumulation_steps",
            1,
        ),
    )


def build_logger(cfg, accelerator):
    """Initialize Weights & Biases on the main process."""
    if not accelerator.is_main_process:
        return

    project = f"elt-{cfg['model']['name']}"
    run_name = (
        f"{build_experiment_name(cfg)}:"
        f"{cfg['model']['variant']}:"
        f"run{cfg['run_number']}"
    )

    wandb.init(
        project=project,
        name=run_name,
        config=cfg,
    )


def build_adamw(params, optim_cfg):
    return torch.optim.AdamW(
        params,
        lr=float(optim_cfg["lr"]),
        betas=tuple(optim_cfg.get("betas", (0.9, 0.999))),
        eps=float(optim_cfg.get("eps", 1e-8)),
        fused=optim_cfg.get("fused", torch.cuda.is_available()),
    )


OPTIMIZER_BUILDERS = {
    "adamw": build_adamw,
}


def build_optimizer(model, cfg):
    optim_cfg = cfg["optimizer"]
    name = optim_cfg["name"]

    if name not in OPTIMIZER_BUILDERS:
        raise ValueError(
            f"Unknown optimizer '{name}', "
            f"expected one of {list(OPTIMIZER_BUILDERS)}"
        )

    params = build_param_groups(
        model,
        optim_cfg["weight_decay"],
    )

    return OPTIMIZER_BUILDERS[name](params, optim_cfg)


def build_constant_schedule(optimizer, cfg):
    return None


def build_linear_warmup_cosine_schedule(optimizer, cfg):
    total_steps = cfg["train"]["total_steps"]
    warmup_ratio = float(cfg["scheduler"].get("warmup_ratio", 0.0))
    warmup_steps = int(warmup_ratio * total_steps)
    min_lr_ratio = float(cfg["scheduler"].get("min_lr_ratio", 0.0))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)

        progress = (step - warmup_steps) / max(
            1,
            total_steps - warmup_steps,
        )
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_linear_warmup_constant_schedule(optimizer, cfg):
    total_steps = cfg["train"]["total_steps"]
    warmup_ratio = float(cfg["scheduler"].get("warmup_ratio", 0.0))
    warmup_steps = int(warmup_ratio * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


SCHEDULER_BUILDERS = {
    "constant": build_constant_schedule,
    "linear_warmup_cosine": build_linear_warmup_cosine_schedule,
    "linear_warmup_constant": build_linear_warmup_constant_schedule,
}


def build_scheduler(optimizer, cfg):
    name = cfg["scheduler"]["name"]

    if name not in SCHEDULER_BUILDERS:
        raise ValueError(
            f"Unknown scheduler '{name}', "
            f"expected one of {list(SCHEDULER_BUILDERS)}"
        )

    return SCHEDULER_BUILDERS[name](optimizer, cfg)


def maybe_resume(
    cfg,
    model,
    optimizer=None,
    scheduler=None,
    ema=None,
    device="cpu",
):
    resume = cfg["train"].get("resume")

    if resume is None:
        return 0

    if resume == "latest":
        checkpoint_dir = cfg["train"].get(
            "checkpoints_dir",
            "checkpoints",
        )
        resume = get_latest_checkpoint(checkpoint_dir)

        if resume is None:
            print("No checkpoint found, starting from step 0")
            return 0

    step = load_checkpoint(
        resume,
        model,
        optimizer,
        scheduler=scheduler,
        ema=ema,
        device=device,
    )

    print(f"Resumed from {resume} (step {step})")
    return step


def get_latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None

    checkpoints = []

    for filename in os.listdir(directory):
        if not filename.endswith(".pt"):
            continue

        try:
            step = int(
                filename.split("_")[-1].replace(".pt", "")
            )
            checkpoints.append((step, filename))
        except ValueError:
            continue

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])

    return os.path.join(
        directory,
        checkpoints[-1][1],
    )


def count_parameters(model, trainable_only=False):
    if trainable_only:
        return sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

    return sum(p.numel() for p in model.parameters())


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True


def build_param_groups(model, weight_decay):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {
            "params": decay,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay,
            "weight_decay": 0.0,
        },
    ]


def move_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)

    if isinstance(batch, (list, tuple)):
        return type(batch)(
            move_to_device(x, device)
            for x in batch
        )

    if isinstance(batch, dict):
        return {
            k: move_to_device(v, device)
            for k, v in batch.items()
        }

    return batch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model, step=None):
        if step is not None:
            decay = min(
                self.decay,
                (1 + step) / (10 + step),
            )
        else:
            decay = self.decay

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            shadow = self.shadow[name]

            if shadow.device != param.device:
                shadow = shadow.to(param.device)
                self.shadow[name] = shadow

            shadow.mul_(decay)
            shadow.add_(param.data, alpha=1 - decay)

    def apply_shadow(self, model):
        backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

        return backup

    def restore(self, model, backup):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(backup[name])


def save_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    ema=None,
    epoch=0,
    cfg=None,
    scaling_factor=None,
):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()

    if ema is not None:
        checkpoint["ema"] = ema.shadow

    if cfg is not None:
        checkpoint["config"] = cfg

    if scaling_factor is not None:
        checkpoint["scaling_factor"] = scaling_factor

    torch.save(checkpoint, path)


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    ema=None,
    device="cpu",
):
    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    if ema is not None and "ema" in checkpoint:
        ema.shadow = checkpoint["ema"]

    return checkpoint["epoch"]


class InfiniteDataLoader:
    def __init__(self, dataloader):
        self.loader = dataloader
        self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def build_vae_from_checkpoint(path, device="cpu", freeze=True):
    checkpoint = torch.load(
        path,
        map_location=device,
    )

    cfg = checkpoint["config"]

    vae = build_model(cfg)
    vae.load_state_dict(checkpoint["model"])
    vae.to(device)
    vae.eval()

    if freeze:
        freeze_model(vae)

    return vae


def denormalize(x):
    return ((x + 1) / 2).clamp(0, 1)


def sample_labels(
    num_classes,
    num_images,
    device,
    mode="random",
):
    if num_classes is None:
        return None

    if mode == "cycle":
        return torch.arange(
            num_images,
            device=device,
        ) % num_classes

    return torch.randint(
        0,
        num_classes,
        (num_images,),
        device=device,
    )


class ELTSchedule:
    def __init__(self, cfg, total_steps):
        self.name = cfg["schedule"]["name"]
        self.lambda_max = cfg["lambda"]

        if self.name not in ("constant", "linear_warmup"):
            raise ValueError(
                f"Unknown ELT schedule '{self.name}', "
                "expected 'constant' or 'linear_warmup'"
            )

        if self.name == "linear_warmup":
            self.warmup_steps = int(
                cfg["schedule"]["warmup_ratio"] * total_steps
            )

    def __call__(self, step):
        if self.name == "constant":
            return self.lambda_max

        if self.name == "linear_warmup":
            if step >= self.warmup_steps:
                return self.lambda_max

            return self.lambda_max * step / self.warmup_steps

        raise ValueError(
            f"Unknown ELT schedule '{self.name}', "
            "expected 'constant' or 'linear_warmup'"
        )


def sample_intermediate_loops(cfg, loop_steps):
    strategy = cfg["elt"]["strategy"]
    lmax = loop_steps

    if strategy == "fixed":
        return [
            int(x)
            for x in cfg["elt"]["intermediate_loops"]
        ] + [lmax]

    if strategy == "random":
        l_min = int(cfg["elt"]["l_min"])

        if lmax - 1 < l_min:
            raise ValueError(
                f"l_min ({l_min}) leaves no valid intermediate "
                f"range for loop_steps={lmax}"
            )

        lint = random.randint(l_min, lmax - 1)
        return [lint, lmax]

    raise ValueError(f"Unknown ELT strategy: {strategy}")


def build_distillation(cfg):
    if not cfg["elt"]["enabled"]:
        return None

    name = cfg["elt"]["distillation"].get("type")

    if name is None:
        return None

    if name == "mse":
        return F.mse_loss

    if name == "smooth_l1":
        return F.smooth_l1_loss

    raise ValueError(
        f"Unknown distillation loss: {name}"
    )

def build_repa(cfg, model):
    if not cfg["repa"]["enabled"]:
        return None, None

    encoder_name = cfg["repa"]["encoder"]

    if encoder_name == "dinov2":
        encoder = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14",
        )
        repr_dim = encoder.embed_dim
    else:
        raise ValueError(f"Unknown REPA encoder '{encoder_name}'")

    dit_dim = model.cfg.hidden_size

    repa = REPA(
        dit_dim=dit_dim,
        repr_dim=repr_dim,
    )

    return repa, encoder

def compute_latent_scaling_factor(cfg, split="train"):
    files = sorted(glob.glob(
        os.path.join(cfg[f"latent_cache_path_{split}"], "chunk_*.pt")
    ))

    if not files:
        raise FileNotFoundError("No latent cache found.")

    total = 0
    sum_sq = 0.0

    for f in files:
        latents = torch.load(
            f,
            map_location="cpu",
            weights_only=True,
        )["latents"]

        sum_sq += latents.double().pow(2).sum().item()
        total += latents.numel()

    std = (sum_sq / total) ** 0.5
    return 1.0 / std