"""Package the report, experiment code, and recorded results."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "output/pdf/hw2_wikiart_gan_report_nautilus.pdf"
OUTPUT = ROOT / "output/hw2_submission_nautilus.zip"


def main():
    # Include reproducible code and evidence; omit raw images and optimizer states.
    files = [
        "requirements.txt", "models.py", "data.py", "prepare_data.py", "train.py",
        "evaluate.py", "generate.py", "audit_data.py", "audit_results.py",
        "plot_results.py", "nautilus/build_final.py", "nautilus/package_final.py",
        "nautilus/run.sh", "nautilus/job.yaml", "nautilus/README.md",
        "nautilus/inputs.sha256", "nautilus/inputs.original.sha256",
        "results/main/generator_020.pt",
        "results/main/config.json", "results/main/history.csv", "results/main/completed.json",
        "data/source.json", "data/wikiart64/summary.json", "data/wikiart64/manifest.csv",
        "data/wikiart64/audit.json", "results/nautilus-v1/selection.json",
        "results/nautilus-v1/completed",
    ]
    files += [str(p.relative_to(ROOT)) for p in (ROOT / "nautilus/payload").glob("*.py")]
    files += [str(p.relative_to(ROOT)) for p in (ROOT / "report_nautilus").rglob("*") if p.is_file()]
    for model in ("plain", "resize-sn"):
        for name in ("config.json", "completed.json", "history.csv", "generator_040.pt", "epoch_040.png"):
            files.append(f"results/nautilus-v1/{model}/{name}")
    for directory in ("results/evaluation", "results/nautilus-v1/original_validation",
                      "results/nautilus-v1/plain_validation", "results/nautilus-v1/resize-sn_validation",
                      "results/nautilus-v1/plain_test"):
        for name in ("metrics.json", "audit.json", "generated_grid.png", "real_grid.png"):
            files.append(f"{directory}/{name}")
    # Support both the development workspace and a fresh repository checkout.
    pdf = PDF if PDF.is_file() else ROOT / "report.pdf"
    readme = ROOT / "nautilus/FINAL_README.md"
    if not readme.is_file():
        readme = ROOT / "README.md"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUTPUT, "w", ZIP_DEFLATED, compresslevel=5) as archive:
        archive.write(pdf, "report.pdf")
        archive.write(readme, "README.md")
        for path in files:
            archive.write(ROOT / path, path)
    # Check archive integrity and ensure its PDF is the current final report.
    with ZipFile(OUTPUT) as archive:
        assert archive.testzip() is None
        assert archive.read("report.pdf") == pdf.read_bytes()
    print(OUTPUT)


if __name__ == "__main__":
    main()
