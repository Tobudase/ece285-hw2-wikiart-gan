"""Verify the completed comparison and build its two-page NeurIPS report."""
import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/nautilus-v1"
REPORT = ROOT / "report_nautilus"
OUTPUT = ROOT / "output/pdf/hw2_wikiart_gan_report_nautilus.pdf"


def read(path):
    return json.loads(path.read_text())


def checked_metrics(directory, checkpoint):
    m = read(directory / "metrics.json")
    a = read(directory / "audit.json")
    # Accept only audited scores tied to the checkpoint included in the submission.
    assert a["passed"]
    assert m["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for key, value in a["independently_recomputed"].items():
        assert math.isclose(m[key], value, abs_tol=1e-5, rel_tol=1e-5)
    return m


def training(name):
    directory = RUN / name
    cfg = read(directory / "config.json")
    done = read(directory / "completed.json")
    with (directory / "history.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    # Require a complete, consecutive history for the declared training budget.
    assert cfg["epochs"] == done["epochs"] == 40
    assert [int(row["epoch"]) for row in rows] == list(range(1, 41))
    for row in rows:
        assert all(math.isfinite(float(row[key])) for key in ("discriminator_loss", "generator_loss", "seconds"))
        assert int(row["examples"]) == cfg["train_count"] // cfg["batch_size"] * cfg["batch_size"]
    return cfg, done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tectonic", default=shutil.which("tectonic"))
    args = parser.parse_args()
    assert (RUN / "completed").is_file()
    cfg, wide = training("plain")
    _, resize = training("resize-sn")
    baseline = ROOT / "results/main/generator_020.pt"
    final = RUN / "plain/generator_040.pt"
    base_val = checked_metrics(RUN / "original_validation", baseline)
    wide_val = checked_metrics(RUN / "plain_validation", final)
    resize_val = checked_metrics(RUN / "resize-sn_validation", RUN / "resize-sn/generator_040.pt")
    base_test = checked_metrics(ROOT / "results/evaluation", baseline)
    wide_test = checked_metrics(RUN / "plain_test", final)
    for value, split, epoch in ((base_val, "validation", 20), (wide_val, "validation", 40),
                                (resize_val, "validation", 40), (base_test, "test", 20),
                                (wide_test, "test", 40)):
        assert value["split"] == split and value["epoch"] == epoch
        for key in ("manifest_sha256", "seed", "latent_seed", "samples"):
            assert value[key] == wide_test[key]
    # Confirm selection used validation FID before reporting held-out test results.
    scores = {"original": base_val["fid_2048"], "plain": wide_val["fid_2048"],
              "resize-sn": resize_val["fid_2048"]}
    assert read(RUN / "selection.json") == {"winner": "plain", "validation_fid": scores}
    assert min(scores, key=scores.get) == "plain"
    assert cfg["manifest_sha256"] == wide_test["manifest_sha256"]
    data = read(ROOT / "data/wikiart64/summary.json")
    n = lambda x: f"{x:,}"
    # Populate report numbers from verified records instead of copying them by hand.
    macros = {
        "RawCount": n(data["candidate_images"]), "UniqueCount": n(data["unique_images"]),
        "DuplicateCount": n(data["duplicates"]), "StyleCount": len(data["styles"]),
        "TrainCount": n(data["counts"]["train"]), "ValCount": n(data["counts"]["validation"]),
        "TestCount": n(data["counts"]["test"]), "EpochCount": cfg["epochs"],
        "BatchSize": cfg["batch_size"], "LatentDim": cfg["latent_dim"], "Width": cfg["width"],
        "TrainingSeed": cfg["seed"], "RealSeed": wide_test["seed"], "LatentSeed": wide_test["latent_seed"],
        "GeneratorParameters": n(cfg["generator_parameters"]),
        "DiscriminatorParameters": n(cfg["discriminator_parameters"]),
        "BatchesPerEpoch": cfg["train_count"] // cfg["batch_size"],
        "WideMinutes": f'{wide["elapsed_seconds"]/60:.2f}', "ResizeMinutes": f'{resize["elapsed_seconds"]/60:.2f}',
        "EvaluationCount": n(wide_test["samples"]),
        "TestFidProb": f'{wide_test["fid_1000"]:.4f}', "TestRealIS": f'{wide_test["is_real"]:.3f}',
        "TestReduction": f'{100*(1-wide_test["fid_2048"]/base_test["fid_2048"]):.1f}',
    }
    for name, value in (("ValBase", base_val), ("ValWide", wide_val), ("ValResize", resize_val),
                        ("TestBase", base_test), ("TestWide", wide_test)):
        macros[name + "Fid"] = f'{value["fid_2048"]:.2f}'
        macros[name + "IS"] = f'{value["is_generated"]:.3f}'
    (REPORT / "metrics.tex").write_text("".join(r"\newcommand{\%s}{%s}" % pair + "\n" for pair in macros.items()))
    subprocess.run([sys.executable, str(ROOT / "plot_results.py"), "--run", str(RUN / "plain")], check=True)
    for source, name in ((RUN / "plain/loss_curve.pdf", "loss_curve.pdf"),
                         (ROOT / "results/evaluation/generated_grid.png", "baseline.png"),
                         (RUN / "plain_test/generated_grid.png", "wide.png"),
                         (RUN / "plain_test/real_grid.png", "real.png")):
        shutil.copy2(source, REPORT / "images" / name)
    if not args.tectonic:
        parser.error("provide --tectonic or install tectonic")
    stage = ROOT / "tmp/pdfs/nautilus"
    stage.mkdir(parents=True, exist_ok=True)
    subprocess.run([args.tectonic, "--keep-logs", "--outdir", str(stage), "main.tex"], cwd=REPORT, check=True)
    info = subprocess.check_output(["pdfinfo", str(stage / "main.pdf")], text=True)
    # Do not replace the deliverable unless the compiled report has exactly two pages.
    if not re.search(r"^Pages:\s+2\s*$", info, re.M):
        raise ValueError("Expected exactly two pages; inspect staged PDF")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stage / "main.pdf", OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
