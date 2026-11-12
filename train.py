"""Train on the training partition only and save auditable epoch records."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data import WikiArtCache
from models import Generator, Discriminator, initialize, adversarial_step


def select_device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/wikiart64"))
    parser.add_argument("--output", type=Path, default=Path("results/main"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=285)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.width < 1 or args.latent_dim < 1 or args.lr <= 0:
        parser.error("epochs, width, latent dimension and lr must be positive; batch size >= 2")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    dataset = WikiArtCache(args.data, "train")
    if len(dataset) < args.batch_size:
        raise ValueError("Not enough training images for one full batch")
    if dataset.images.shape[1:3] != (64, 64):
        raise ValueError("This architecture requires a 64 x 64 cache")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "last.pt"
    if checkpoint_path.exists() and not args.resume:
        raise FileExistsError("Existing run: use --resume or choose a fresh output directory")
    # Separate random streams keep data order independent of latent sampling.
    loader_rng = torch.Generator().manual_seed(args.seed)
    noise_rng = torch.Generator().manual_seed(args.seed + 1)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=0, generator=loader_rng)
    generator = Generator(args.latent_dim, args.width).to(device)
    discriminator = Discriminator(args.width).to(device)
    generator.apply(initialize)
    discriminator.apply(initialize)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    # Reuse preview noise across epochs so visual changes reflect model updates.
    fixed_z = torch.randn(64, args.latent_dim, generator=torch.Generator().manual_seed(args.seed + 2)).to(device)
    # Bind the run to its exact dataset manifest for later evaluation checks.
    manifest_hash = hashlib.sha256((args.data / "manifest.csv").read_bytes()).hexdigest()
    configuration = {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items() if key != "resume"}
    configuration.update(device_used=str(device), torch_version=str(torch.__version__),
                         platform=platform.platform(), train_count=len(dataset),
                         manifest_sha256=manifest_hash, optimizer_betas=[0.5, 0.999],
                         generator_parameters=sum(p.numel() for p in generator.parameters()),
                         discriminator_parameters=sum(p.numel() for p in discriminator.parameters()),
                         checkpoint_policy="Final epoch of a predeclared budget; no test-based selection",
                         dropped_per_epoch=len(dataset) % args.batch_size)
    history = []
    start_epoch, previous_seconds = 1, 0.0
    if args.resume:
        saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
        for key in ("width", "latent_dim", "seed", "lr", "batch_size", "manifest_sha256", "device_used"):
            if configuration[key] != saved["config"][key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        if args.epochs < saved["epoch"]:
            raise ValueError("Requested epochs cannot precede the saved checkpoint epoch")
        if args.epochs == saved["epoch"]:
            print(f"Checkpoint already completed epoch {saved['epoch']}; no changes made.")
            return
        generator.load_state_dict(saved["generator"])
        discriminator.load_state_dict(saved["discriminator"])
        opt_g.load_state_dict(saved["optimizer_g"])
        opt_d.load_state_dict(saved["optimizer_d"])
        # Restore sampling streams as well as model and optimizer states.
        loader_rng.set_state(saved["loader_rng"].cpu())
        noise_rng.set_state(saved["noise_rng"].cpu())
        history = saved["history"]
        start_epoch = saved["epoch"] + 1
        previous_seconds = saved["elapsed_seconds"]
    # An extended run must not retain the previous budget's completion marker.
    (args.output / "completed.json").unlink(missing_ok=True)
    (args.output / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        generator.train()
        discriminator.train()
        sums = torch.zeros(4, device=device)
        count = 0
        epoch_start = time.perf_counter()
        for step, real in enumerate(loader, 1):
            real = real.to(device)
            z = torch.randn(real.size(0), args.latent_dim, generator=noise_rng).to(device)
            values = adversarial_step(generator, discriminator, real, z, opt_g, opt_d)
            # Weight batch means by sample count to obtain epoch means.
            sums += values * real.size(0)
            count += real.size(0)
            if step % 100 == 0:
                print(f"epoch={epoch} step={step}/{len(loader)} losses={values[:2].cpu().tolist()}", flush=True)
        means = (sums / count).cpu().tolist()
        if not all(torch.isfinite(torch.tensor(means))):
            raise FloatingPointError("Non-finite training losses")
        row = dict(epoch=epoch, discriminator_loss=means[0], generator_loss=means[1],
                   d_real_probability=means[2], d_fake_probability=means[3],
                   examples=count, seconds=time.perf_counter() - epoch_start)
        history.append(row)
        with (args.output / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(history)
        # Preview generation must not update BatchNorm running statistics.
        generator.eval()
        with torch.no_grad():
            preview = generator(fixed_z).cpu()
        save_image((preview + 1) / 2, args.output / f"epoch_{epoch:03d}.png", nrow=8, padding=2)
        saved = dict(generator=generator.state_dict(), discriminator=discriminator.state_dict(),
                     optimizer_g=opt_g.state_dict(), optimizer_d=opt_d.state_dict(),
                     config=configuration, history=history, epoch=epoch,
                     loader_rng=loader_rng.get_state(), noise_rng=noise_rng.get_state(),
                     elapsed_seconds=previous_seconds + time.perf_counter() - started)
        # Replace the checkpoint atomically to avoid exposing a partial save.
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(saved, temporary)
        os.replace(temporary, checkpoint_path)
        if epoch in {1, 5, 10, 20} or epoch == args.epochs:
            torch.save(dict(generator=generator.state_dict(), config=configuration, epoch=epoch),
                       args.output / f"generator_{epoch:03d}.pt")
        print(json.dumps(row), flush=True)
    (args.output / "completed.json").write_text(json.dumps(dict(
        epochs=history[-1]["epoch"], elapsed_seconds=previous_seconds + time.perf_counter() - started,
        checkpoint=str(checkpoint_path), final=history[-1]), indent=2) + "\n")


if __name__ == "__main__":
    main()
