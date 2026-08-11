"""
Dataset and DataLoader utilities for ELT / ViT-VAE.
"""

import glob
import io

import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Sampler
import random

def normalize_to_neg_one_to_one(x: torch.Tensor) -> torch.Tensor:
    """Map tensor values from [0, 1] to [-1, 1]."""
    return x * 2.0 - 1.0


class ShardShuffleSampler(Sampler):
    """
    Shuffles shard order each epoch, and row order within each shard,
    but keeps consecutive indices grouped by shard -- so a single-file
    LRU cache (like ImageNetParquetDataset._load_table) stays effective,
    instead of being defeated by a naive fully-random shuffle.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        # group sample indices by which file they belong to
        self.shard_indices = {}
        for i, (file, _) in enumerate(dataset.samples):
            self.shard_indices.setdefault(file, []).append(i)

    def __iter__(self):
        shard_order = list(self.shard_indices.keys())
        random.shuffle(shard_order)
        for shard in shard_order:
            indices = self.shard_indices[shard][:]
            random.shuffle(indices)
            yield from indices

    def __len__(self):
        return len(self.dataset)

class ImageOnlyDataset(Dataset):
    """Wrapper that returns only images from an image-label dataset."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0]


class LatentDataset(Dataset):
    """Dataset of precomputed latents with optional labels."""

    def __init__(self, latents: torch.Tensor, labels=None):
        assert isinstance(latents, torch.Tensor)
        if labels is not None:
            assert len(labels) == len(latents)
        self.latents = latents
        self.labels = labels

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        z = self.latents[idx]
        if self.labels is None:
            return z
        return z, self.labels[idx]


class ImageNetParquetDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        pattern = f"{root}/{split}-*.parquet"
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No parquet files found: {pattern}")
        self.samples = []
        self._table = None
        self._table_file = None
        self._file_row_counts = []
        for file in self.files:
            metadata = pq.ParquetFile(file).metadata
            rows = metadata.num_rows
            self._file_row_counts.append(rows)
            self.samples.extend((file, i) for i in range(rows))

    def __len__(self):
        return len(self.samples)

    def _load_table(self, file):
        if self._table_file != file:
            self._table = pq.read_table(file, columns=["image", "label"])
            self._table_file = file
        return self._table

    def __getitem__(self, idx):
        file, row_idx = self.samples[idx]
        table = self._load_table(file)
        sample = table["image"][row_idx].as_py()
        label = table["label"][row_idx].as_py()
        image = Image.open(io.BytesIO(sample["bytes"])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def _make_loader(dataset, cfg, is_train):
    """Create a DataLoader with the project's standard settings."""
    num_workers = cfg["num_workers"]

    if is_train and isinstance(dataset, ImageNetParquetDataset):
        return DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            sampler=ShardShuffleSampler(dataset),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
        )

    return DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=is_train, drop_last=is_train,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )


def extract_images(batch):
    """Extract images from supported batch formats."""

    if torch.is_tensor(batch):
        return batch

    if isinstance(batch, (tuple, list)):
        return batch[0]

    if isinstance(batch, dict):
        if "image" in batch:
            return batch["image"]
        if "images" in batch:
            return batch["images"]

        raise KeyError(
            f"Cannot find image key. Available keys: {batch.keys()}"
        )

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def extract_labels(batch):
    """Extract labels from supported batch formats."""

    if isinstance(batch, (tuple, list)):
        return batch[1]

    if isinstance(batch, dict):
        if "label" in batch:
            return batch["label"]
        if "labels" in batch:
            return batch["labels"]

    return None


def build_dataloader(cfg, split="train"):
    """Build an image DataLoader from the data configuration."""

    transform = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        normalize_to_neg_one_to_one,
    ])

    dataset_name = cfg["dataset"].lower()

    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(
            root=cfg["root"],
            train=(split == "train"),
            transform=transform,
            download=cfg.get("download", True),
        )

    elif dataset_name == "imagefolder":
        dataset = datasets.ImageFolder(
            root=f"{cfg['root']}/{split}",
            transform=transform,
        )

    elif dataset_name == "imagenet_parquet":
        dataset = ImageNetParquetDataset(
            root=cfg["root"],
            split=split,
            transform=transform,
        )

    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    return _make_loader(
        dataset,
        cfg,
        is_train=(split == "train"),
    )


def build_latent_dataloader(cfg, split="train"):
    """Build a DataLoader from precomputed latent tensors."""

    latents = torch.load(
        cfg[f"latent_cache_path_{split}"]
    )

    label_key = f"label_cache_path_{split}"
    labels = (
        torch.load(cfg[label_key])
        if cfg.get(label_key)
        else None
    )

    dataset = LatentDataset(latents, labels)

    return _make_loader(
        dataset,
        cfg,
        is_train=(split == "train"),
    )


@torch.no_grad()
def build_latent_cache(encoder, cfg, split="train", device=None):
    """Encode an entire dataset split and save its latent representations."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    image_loader = build_dataloader(cfg, split=split)

    was_training = encoder.training
    encoder.eval()
    encoder.to(device)

    latent_list = []

    try:
        for batch in image_loader:
            images = extract_images(batch).to(
                device,
                non_blocking=True,
            )
            mu, _ = encoder(images)
            latent_list.append(mu.cpu())
    finally:
        encoder.train(was_training)

    latents = torch.cat(latent_list, dim=0)

    save_path = cfg[f"latent_cache_path_{split}"]
    torch.save(latents, save_path)

    return latents


@torch.no_grad()
def compute_scaling_factor(vae, cfg, split="train", device=None):
    """Compute the latent scaling factor from a dataset split."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    image_loader = build_dataloader(
        cfg["data"],
        split=split,
    )

    was_training = vae.encoder.training
    vae.encoder.eval()
    vae.encoder.to(device)

    latent_list = []

    try:
        for batch in image_loader:
            images = extract_images(batch).to(
                device,
                non_blocking=True,
            )
            mu, _ = vae.encoder(images)
            latent_list.append(mu.cpu())
    finally:
        vae.encoder.train(was_training)

    latents = torch.cat(latent_list, dim=0)

    return (1.0 / latents.std()).item()