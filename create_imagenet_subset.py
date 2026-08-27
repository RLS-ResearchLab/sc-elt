import glob
import random
import torch
import pyarrow.parquet as pq

ROOT = "./data/imagenet/data"
OUTPUT = "./data/imagenet_200k_indices.pt"

SEED = 42
PER_CLASS = 200

files = sorted(glob.glob(f"{ROOT}/train-*.parquet"))

class_indices = {}

global_index = 0

for file in files:
    table = pq.read_table(file, columns=["label"])
    labels = table["label"].to_pylist()

    for label in labels:
        class_indices.setdefault(label, []).append(global_index)
        global_index += 1

random.seed(SEED)

selected = []

for label in sorted(class_indices):
    indices = class_indices[label]

    if len(indices) < PER_CLASS:
        raise RuntimeError(
            f"Class {label} has only {len(indices)} images"
        )

    selected.extend(
        random.sample(indices, PER_CLASS)
    )

random.shuffle(selected)

indices = torch.tensor(selected, dtype=torch.long)

torch.save(indices, OUTPUT)

print(f"Total ImageNet samples: {global_index}")
print(f"Selected samples: {len(indices)}")
print(f"Saved to: {OUTPUT}")