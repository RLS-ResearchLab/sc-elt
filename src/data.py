"""
Dataset and DataLoader utilities for ELT / ViT-VAE.
"""

import glob
import io
import os
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

class ChunkedLatentDataset(Dataset):
    def __init__(self, cache_dir):
        self.files = sorted(
            glob.glob(
                os.path.join(cache_dir, "chunk_*.pt")
            )
        )

        if not self.files:
            raise FileNotFoundError(
                f"No latent chunks found in {cache_dir}"
            )

        self.sizes = [
            torch.load(
                f,
                map_location="cpu",
                weights_only=True,
            )["latents"].shape[0]
            for f in self.files
        ]

        self.cumulative = torch.tensor(
            self.sizes
        ).cumsum(0)

        self.cache = None
        self.cache_idx = None

    def __len__(self):
        return int(self.cumulative[-1])

    def __getitem__(self, idx):
        chunk_idx = int(
            torch.searchsorted(
                self.cumulative,
                idx,
                right=True,
            )
        )

        prev = (
            0
            if chunk_idx == 0
            else int(self.cumulative[chunk_idx - 1])
        )

        local_idx = idx - prev

        if self.cache_idx != chunk_idx:
            self.cache = torch.load(
                self.files[chunk_idx],
                map_location="cpu",
                weights_only=True,
            )
            self.cache_idx = chunk_idx

        return (
            self.cache["latents"][local_idx],
            self.cache["labels"][local_idx],
        )
class ImageNetParquetDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        transform=None,
        indices_path=None,
    ):
        self.root = root
        self.split = split
        self.transform = transform

        pattern = f"{root}/{split}-*.parquet"
        self.files = sorted(glob.glob(pattern))

        if not self.files:
            raise FileNotFoundError(
                f"No parquet files found: {pattern}"
            )

        self.samples = []
        self._table = None
        self._table_file = None

        for file in self.files:
            rows = pq.ParquetFile(file).metadata.num_rows

            self.samples.extend(
                (file, i)
                for i in range(rows)
            )

        if indices_path is not None:
            indices = torch.load(
                indices_path,
                map_location="cpu",
                weights_only=True,
            )

            self.samples = [
                self.samples[i]
                for i in indices.tolist()
            ]

    def __len__(self):
        return len(self.samples)

    def _load_table(self, file):
        if self._table_file != file:
            self._table = pq.read_table(
                file,
                columns=["image", "label"],
            )
            self._table_file = file

        return self._table

    def __getitem__(self, idx):
        file, row_idx = self.samples[idx]

        table = self._load_table(file)

        sample = table["image"][row_idx].as_py()
        label = table["label"][row_idx].as_py()

        image = Image.open(
            io.BytesIO(sample["bytes"])
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
class FFHQParquetDataset(Dataset):
    """
    FFHQ 256x256 dataset stored as Parquet shards.

    The original dataset contains only a train split.
    We create a deterministic virtual split:

        train -> first 69,000 images
        test  -> last 1,000 images

    No files are copied.
    """

    def __init__(
        self,
        root,
        split="train",
        transform=None,
        val_size=1000,
    ):
        self.root = root
        self.split = split
        self.transform = transform

        pattern = f"{root}/{split if split == 'train' else 'train'}-*.parquet"
        self.files = sorted(glob.glob(pattern))

        if not self.files:
            raise FileNotFoundError(
                f"No FFHQ parquet files found: {pattern}"
            )

        self.samples = []

        self._table = None
        self._table_file = None

        for file in self.files:
            metadata = pq.ParquetFile(file).metadata
            rows = metadata.num_rows

            self.samples.extend(
                (file, i)
                for i in range(rows)
            )

        total = len(self.samples)

        if total <= val_size:
            raise ValueError(
                f"FFHQ dataset has only {total} samples, "
                f"but val_size={val_size}"
            )

        if split == "train":
            self.samples = self.samples[:-val_size]

        elif split == "test":
            self.samples = self.samples[-val_size:]

        else:
            raise ValueError(
                f"Unknown FFHQ split '{split}'. "
                f"Use 'train' or 'test'."
            )

    def __len__(self):
        return len(self.samples)

    def _load_table(self, file):
        if self._table_file != file:
            self._table = pq.read_table(
                file,
                columns=["image"],
            )
            self._table_file = file

        return self._table

    def __getitem__(self, idx):
        file, row_idx = self.samples[idx]

        table = self._load_table(file)

        sample = table["image"][row_idx].as_py()

        image = Image.open(
            io.BytesIO(sample["bytes"])
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image

def _make_loader(dataset, cfg, is_train):
    """Create a DataLoader with the project's standard settings."""
    num_workers = cfg["num_workers"]

    if is_train and isinstance(
        dataset,
        (ImageNetParquetDataset, FFHQParquetDataset),
    ):
        return DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            sampler=ShardShuffleSampler(dataset),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
        )

    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=is_train,
        drop_last=is_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
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
        indices_path = None

        if split == "train":
            indices_path = cfg.get("subset_indices")

        dataset = ImageNetParquetDataset(
            root=cfg["root"],
            split=split,
            transform=transform,
            indices_path=indices_path,
        )
    elif dataset_name == "ffhq_parquet":
        dataset = FFHQParquetDataset(
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
    dataset = ChunkedLatentDataset(
        cfg[f"latent_cache_path_{split}"]
    )

    return _make_loader(
        dataset,
        cfg,
        is_train=(split == "train"),
    )


@torch.no_grad()
def build_latent_cache(
    encoder,
    cfg,
    split="train",
    device=None,
    chunk_size=9984,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_dir = cfg[f"latent_cache_path_{split}"]
    os.makedirs(cache_dir, exist_ok=True)

    files = sorted(
        glob.glob(os.path.join(cache_dir, "chunk_*.pt"))
    )

    if files:
        chunk_id = (
            int(
                os.path.basename(files[-1])
                .split("_")[1]
                .split(".")[0]
            ) + 1
        )

        cached = sum(
            torch.load(
                f,
                map_location="cpu",
                weights_only=True,
            )["latents"].shape[0]
            for f in files
        )
    else:
        chunk_id = 0
        cached = 0

    loader = build_dataloader(cfg, split=split)
    dataset = loader.dataset

    if cached:
        if not hasattr(dataset, "samples"):
            raise RuntimeError(
                "Dataset does not support cache resume."
            )

        dataset.samples = dataset.samples[cached:]

    print(
        f"Resuming from {cached} images, chunk {chunk_id}",
        flush=True,
    )

    encoder.eval().to(device)

    latent_chunks = []
    label_chunks = []
    total = cached

    for batch in loader:
        images = extract_images(batch).to(
            device,
            non_blocking=True,
        )

        labels = extract_labels(batch).cpu().long()

        if labels is None:
            raise RuntimeError(
                f"Expected labels for {split} latent cache, "
                "but the DataLoader returned none."
            )

        mu, _ = encoder(images)

        latent_chunks.append(mu.cpu())
        label_chunks.append(labels.cpu())

        current_size = sum(
            x.size(0) for x in latent_chunks
        )

        if current_size >= chunk_size:
            latents = torch.cat(latent_chunks)
            labels = torch.cat(label_chunks)

            if latents.size(0) != labels.size(0):
                raise RuntimeError(
                    f"Latent/label mismatch: "
                    f"{latents.size(0)} vs {labels.size(0)}"
                )

            torch.save(
                {
                    "latents": latents,
                    "labels": labels,
                },
                os.path.join(
                    cache_dir,
                    f"chunk_{chunk_id:05d}.pt",
                ),
            )

            print(
                f"chunk {chunk_id}: "
                f"latents={latents.shape}, "
                f"labels={labels.shape}",
                flush=True,
            )

            total += latents.size(0)
            latent_chunks.clear()
            label_chunks.clear()
            chunk_id += 1

    if latent_chunks:
        latents = torch.cat(latent_chunks)
        labels = torch.cat(label_chunks)

        if latents.size(0) != labels.size(0):
            raise RuntimeError(
                f"Latent/label mismatch: "
                f"{latents.size(0)} vs {labels.size(0)}"
            )

        torch.save(
            {
                "latents": latents,
                "labels": labels,
            },
            os.path.join(
                cache_dir,
                f"chunk_{chunk_id:05d}.pt",
            ),
        )

        print(
            f"chunk {chunk_id}: "
            f"latents={latents.shape}, "
            f"labels={labels.shape}",
            flush=True,
        )

        total += latents.size(0)
        chunk_id += 1

    print(
        f"Done: {total} images, {chunk_id} chunks",
        flush=True,
    )


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