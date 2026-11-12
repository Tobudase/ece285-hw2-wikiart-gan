"""Evaluate a fixed checkpoint on held-out data using Ignite IS and two FIDs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.utils import save_image

from data import WikiArtCache
from models import Generator
from train import select_device


def inception_input(images01):
    """Resize square RGB samples and apply torchvision's ImageNet normalization."""
    resized = F.interpolate(images01, size=(299, 299), mode="bilinear", align_corners=False)
    mean = resized.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = resized.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    # InceptionModel enables transform_input internally; do not feed [-1, 1] directly.
    return (resized - mean) / std


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/wikiart64"))
    p.add_argument("--checkpoint", type=Path, default=Path("results/main/generator_020.pt"))
    p.add_argument("--output", type=Path, default=Path("results/evaluation"))
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=286)
    p.add_argument("--latent-seed", type=int, default=1286)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--torch-home", default="data/torch")
    args = p.parse_args()
    if args.samples < 2 or args.batch_size < 1:
        p.error("At least 2 samples and a positive batch size are required")
    if (args.output / "metrics.json").exists():
        raise FileExistsError("Evaluation exists; choose another output directory")
    os.environ["TORCH_HOME"] = str(Path(args.torch_home).resolve())
    torch.set_num_threads(4)
    from ignite.metrics import FID, InceptionScore
    from ignite.metrics.gan.utils import InceptionModel
    from pytorch_fid.inception import InceptionV3

    device = select_device(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    # Evaluation must use fresh noise, independent of training and previews.
    if args.latent_seed in {config["seed"] + 1, config["seed"] + 2}:
        raise ValueError("Evaluation latent seed must differ from training noise and preview seeds")
    dataset = WikiArtCache(args.data, args.split)
    manifest_hash = hashlib.sha256((args.data / "manifest.csv").read_bytes()).hexdigest()
    if manifest_hash != config["manifest_sha256"]:
        raise ValueError("Evaluation manifest differs from training")
    if args.samples > len(dataset):
        raise ValueError("Requested more real samples than the held-out partition contains")
    # Sample distinct held-out images with a reproducible selection seed.
    indices = np.random.default_rng(args.seed).choice(len(dataset), args.samples, replace=False)
    generator = Generator(config["latent_dim"], config["width"]).to(device).eval()
    generator.load_state_dict(saved["generator"])
    noise_rng = torch.Generator().manual_seed(args.latent_seed)
    # The two feature extractors have different weights and input contracts.
    probability_model = InceptionModel(return_features=False, device=device).eval()
    pool3_model = InceptionV3(output_blocks=(3,), use_fid_inception=True).to(device).eval()
    # Identity prevents Ignite from extracting features a second time.
    metrics = dict(fid_2048=FID(num_features=2048, feature_extractor=nn.Identity()),
                   fid_1000=FID(num_features=1000, feature_extractor=nn.Identity()),
                   is_generated=InceptionScore(num_features=1000, feature_extractor=nn.Identity()),
                   is_real=InceptionScore(num_features=1000, feature_extractor=nn.Identity()))
    args.output.mkdir(parents=True, exist_ok=True)
    real_features, fake_features, real_probabilities, fake_probabilities = [], [], [], []
    started = time.perf_counter()
    # Models stay in evaluation mode; metric collection needs no gradients.
    with torch.no_grad():
        for start in range(0, args.samples, args.batch_size):
            selected = indices[start:start + args.batch_size]
            real = torch.stack([dataset[int(i)] for i in selected]).to(device)
            z = torch.randn(len(selected), config["latent_dim"], generator=noise_rng).to(device)
            fake = generator(z)
            # Quantize generated samples to the same 8-bit RGB range as real images.
            real01 = (real + 1) / 2
            fake01 = ((fake + 1) * 127.5).round().clamp(0, 255) / 255
            probs_real = probability_model(inception_input(real01)).cpu()
            probs_fake = probability_model(inception_input(fake01)).cpu()
            features_real = pool3_model(real01)[0].flatten(1).cpu()
            features_fake = pool3_model(fake01)[0].flatten(1).cpu()
            metrics["fid_2048"].update((features_fake, features_real))
            metrics["fid_1000"].update((probs_fake, probs_real))
            metrics["is_generated"].update(probs_fake)
            metrics["is_real"].update(probs_real)
            real_features.append(features_real.numpy()); fake_features.append(features_fake.numpy())
            real_probabilities.append(probs_real.numpy()); fake_probabilities.append(probs_fake.numpy())
            # Show the first seeded samples rather than selecting attractive outputs.
            if start == 0:
                save_image(fake01[:16].cpu(), args.output / "generated_grid.png", nrow=4, padding=2, pad_value=1)
                save_image(real01[:16].cpu(), args.output / "real_grid.png", nrow=4, padding=2, pad_value=1)
            if start % (args.batch_size * 10) == 0:
                print(f"evaluated {start + len(selected)}/{args.samples}", flush=True)
    # Retain features so score computations can be audited without retraining.
    np.savez_compressed(args.output / "features.npz", real=np.concatenate(real_features),
                        generated=np.concatenate(fake_features), real_probabilities=np.concatenate(real_probabilities),
                        generated_probabilities=np.concatenate(fake_probabilities), real_indices=indices)
    print("Computing covariance matrix square roots...", flush=True)
    scores = {name: float(metric.compute()) for name, metric in metrics.items()}
    if not all(np.isfinite(v) for v in scores.values()):
        raise FloatingPointError("Non-finite evaluation score")
    record = dict(**scores, samples=args.samples, split=args.split, seed=args.seed,
                  latent_seed=args.latent_seed, training_noise_seed=config["seed"] + 1,
                  epoch=saved["epoch"], checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  manifest_sha256=manifest_hash, device=str(device), elapsed_seconds=time.perf_counter() - started,
                  fid_2048_extractor="pytorch-fid 0.3.0 TensorFlow-compatible Inception weights, pool3",
                  fid_1000_extractor="Ignite torchvision ImageNet Inception-v3 class probabilities",
                  is_definition="Ignite global IS (one aggregate; no split standard deviation)",
                  input_processing="64x64 RGB; generated images quantized to uint8; Inception bilinear resize 299x299",
                  probability_normalization="torchvision ImageNet mean/std before its internal transform_input",
                  sample_selection="First 16 seeded generated/real samples shown; no visual selection")
    (args.output / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
