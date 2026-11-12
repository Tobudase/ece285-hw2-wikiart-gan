"""Plot measured epoch losses without smoothing or cherry-picking epochs."""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=Path("results/main"))
    args = p.parse_args()
    with (args.run / "history.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    fig, ax = plt.subplots(figsize=(6.4, 1.6), layout="constrained")
    # Plot recorded epoch means directly, with no smoothing or omitted epochs.
    epochs = [int(row["epoch"]) for row in rows]
    ax.plot(epochs, [float(r["discriminator_loss"]) for r in rows], label="Discriminator (real + fake)", color="#1764a0")
    ax.plot(epochs, [float(r["generator_loss"]) for r in rows], label="Generator", color="#cf562b")
    ax.set(xlabel="Epoch", ylabel="Mean BCE loss", xlim=(1, max(2, max(epochs))))
    ax.set_xticks(sorted({epochs[0], epochs[-1], *(epoch for epoch in epochs if epoch % 5 == 0)}))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.run / "loss_curve.png", dpi=220)
    fig.savefig(args.run / "loss_curve.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
