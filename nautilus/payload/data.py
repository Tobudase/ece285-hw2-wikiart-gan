"""Read one explicit partition of the prepared WikiArt cache."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class WikiArtCache(Dataset):
    def __init__(self, directory, split):
        if split not in {"train", "validation", "test"}:
            raise ValueError("Choose train, validation, or test")
        root = Path(directory)
        summary = json.loads((root / "summary.json").read_text())
        # Select only rows belonging to the requested data partition.
        with (root / "manifest.csv").open() as stream:
            self.indices = [int(row["index"]) for row in csv.DictReader(stream)
                            if row["split"] == split]
        if len(self.indices) != summary["counts"].get(split, 0):
            raise ValueError("Manifest does not match summary")
        if not self.indices:
            raise ValueError(f"Empty {split} partition")
        # Memory mapping reads samples on demand instead of loading the entire cache.
        self.images = np.load(root / "images.npy", mmap_mode="r", allow_pickle=False)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        # Copy the read-only mapped pixels before converting them to a tensor.
        pixels = np.array(self.images[self.indices[index]], copy=True)
        # Training tensors have RGB channel order and range [-1, 1].
        return torch.from_numpy(pixels).permute(2, 0, 1).float().div_(127.5).sub_(1)
