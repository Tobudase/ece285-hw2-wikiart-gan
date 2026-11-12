"""Build a deduplicated RGB cache directly from the official WikiArt ZIP."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import threading
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageOps

ZIP_READ_LOCK = threading.Lock()


def split_for(digest, seed):
    """Assign identical processed images to a stable 80/10/10 partition."""
    value = int(hashlib.sha256(f"{seed}:{digest}".encode()).hexdigest()[:16], 16)
    fraction = value / 2**64
    return "train" if fraction < 0.8 else "validation" if fraction < 0.9 else "test"


def decode(zipped, member, size):
    try:
        # Serialize ZIP reads; Pillow decoders may seek, which is unsafe when
        # multiple decoders share the same archive stream. Decode bytes in parallel.
        with ZIP_READ_LOCK:
            raw = zipped.read(member)
        with Image.open(io.BytesIO(raw)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = ImageOps.fit(image, (size, size), method=Image.Resampling.BICUBIC)
            return np.asarray(image), None
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        return None, str(exc)


def decoded_members(zipped, members, size, workers):
    # Bounded batches prevent decoded images from accumulating in memory.
    # executor.map preserves sorted input order, so workers do not change splits.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(members), 128):
            batch = members[start:start + 128]
            yield from zip(batch, executor.map(lambda member: decode(zipped, member, size), batch))


def prepare(archive, output, size=64, seed=285, workers=6):
    if size < 8:
        raise ValueError("Image size must be at least 8")
    # Refuse overwriting a cache that may already support an experiment.
    output.mkdir(parents=True, exist_ok=False)
    with ZipFile(archive) as zipped:
        members = sorted((m for m in zipped.infolist()
                          if not m.is_dir() and Path(m.filename).suffix.lower()
                          in {".jpg", ".jpeg", ".png"}), key=lambda m: m.filename)
        if not members:
            raise ValueError("No images in the archive")
        # Keep one disk-backed array instead of 80,000 individual cache files.
        cache = np.lib.format.open_memmap(output / "images.npy", mode="w+",
                                         dtype=np.uint8, shape=(len(members), size, size, 3))
        seen, counts, styles = {}, {}, {}
        failures, duplicates = [], []
        n = 0
        with (output / "manifest.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["index", "source", "style", "split", "pixel_sha256"])
            writer.writeheader()
            for position, (member, (pixels, error)) in enumerate(decoded_members(zipped, members, size, workers), 1):
                if error is not None:
                    failures.append({"source": member.filename, "error": error})
                    continue
                # Deduplicate the exact pixels seen by the model, not filenames.
                digest = hashlib.sha256(pixels.tobytes()).hexdigest()
                if digest in seen:
                    duplicates.append({"source": member.filename, "duplicate_of": seen[digest]})
                    continue
                seen[digest] = member.filename
                # Content-based assignment keeps the split independent of archive order.
                partition = split_for(digest, seed)
                style = Path(member.filename).parent.name
                cache[n] = pixels
                writer.writerow(dict(index=n, source=member.filename, style=style,
                                     split=partition, pixel_sha256=digest))
                counts[partition] = counts.get(partition, 0) + 1
                styles.setdefault(style, {"train": 0, "validation": 0, "test": 0})[partition] += 1
                n += 1
                if position % 1000 == 0:
                    print(f"Processed {position}/{len(members)}; retained {n}", flush=True)
        cache.flush()
        del cache
    (output / "skipped.json").write_text(json.dumps(failures, indent=2) + "\n")
    (output / "duplicates.json").write_text(json.dumps(duplicates, indent=2) + "\n")
    summary = dict(source="steubk/wikiart",
                   archive=str(archive.resolve()), image_size=size, seed=seed,
                   preprocessing="EXIF transpose; RGB; centered square crop; bicubic resize",
                   split_rule="SHA-256(seed:pixel_sha256), thresholds 0.8 and 0.9",
                   candidate_images=len(members), unique_images=n, counts=counts,
                   skipped=len(failures), duplicates=len(duplicates), styles=styles)
    if not n:
        raise ValueError("No valid images; cache is incomplete")
    # This file marks successful completion; readers reject incomplete caches.
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=285)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.output, args.size, args.seed, args.workers), indent=2))
