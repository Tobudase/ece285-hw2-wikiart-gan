"""Verify every cached image hash and all partition/duplicate accounting."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/wikiart64"))
    args = p.parse_args()
    summary = json.loads((args.data / "summary.json").read_text())
    images = np.load(args.data / "images.npy", mmap_mode="r", allow_pickle=False)
    with (args.data / "manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == summary["unique_images"]
    assert len(rows) + summary["duplicates"] + summary["skipped"] == summary["candidate_images"]
    assert dict(Counter(row["split"] for row in rows)) == summary["counts"]
    # Global uniqueness also rules out exact processed duplicates across splits.
    assert len({row["pixel_sha256"] for row in rows}) == len(rows)
    assert images.dtype == np.uint8 and images.shape[1:] == (summary["image_size"], summary["image_size"], 3)
    # Verify cached pixels against the manifest, not just the reported counts.
    for index, row in enumerate(rows):
        assert int(row["index"]) == index
        assert hashlib.sha256(images[index].tobytes()).hexdigest() == row["pixel_sha256"]
    result = dict(passed=True, images_verified=len(rows), split_counts=summary["counts"],
                  exact_duplicates_across_splits=0,
                  limitation="Pixel hash does not detect approximate duplicates or shared artists")
    (args.data / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
