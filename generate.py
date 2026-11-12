"""Generate a seeded image grid without downloading the dataset or evaluators."""

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from models import Generator
from train import select_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("results/main/generator_020.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/generated.png"))
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1286)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    if args.samples < 1 or args.columns < 1:
        parser.error("samples and columns must be positive")
    if args.output.exists():
        raise FileExistsError("Choose a new output path to preserve existing samples")
    torch.set_num_threads(4)
    device = select_device(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    # Restore the trained architecture and freeze BatchNorm statistics for inference.
    model = Generator(config["latent_dim"], config["width"]).to(device).eval()
    model.load_state_dict(saved["generator"])
    rng = torch.Generator().manual_seed(args.seed)
    images = []
    with torch.no_grad():
        # Small batches keep generation usable on machines with limited memory.
        for start in range(0, args.samples, 32):
            noise = torch.randn(min(32, args.samples - start), config["latent_dim"], generator=rng)
            generated = model(noise.to(device))
            # Match evaluation by converting Tanh output to 8-bit RGB values.
            images.append(((generated + 1) * 127.5).round().clamp(0, 255).cpu() / 255)
    pixels = torch.cat(images)
    if not bool(torch.isfinite(pixels).all()):
        raise FloatingPointError("Generated pixels are not finite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(pixels, args.output, nrow=args.columns, padding=2, pad_value=1)
    print(args.output)


if __name__ == "__main__":
    main()
