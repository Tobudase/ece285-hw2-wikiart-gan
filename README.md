# ECE 285 HW2: WikiArt GAN (Nautilus)

Code repository: https://github.com/Tobudase/ece285-hw2-wikiart-gan

Submit report.pdf to Gradescope. It is the two-page NeurIPS report for the
completed Nautilus comparison. The winner is the wider DCGAN (width 64, epoch
40), selected by validation FID and then evaluated on the held-out test set.

## Code guide

- `nautilus/payload/models.py`: final DCGAN and resize-convolution variant,
  discriminator, initialization, and alternating adversarial updates.
- `nautilus/payload/train.py`: CUDA training, seeded sampling, checkpointing,
  and epoch histories.
- `nautilus/payload/evaluate.py`: held-out FID and Inception Score evaluation.
- `prepare_data.py` and `data.py`: preprocessing, exact-pixel deduplication,
  partition assignment, and cached image loading.
- `generate.py`: standalone sample generation for the selected plain DCGAN.
- `audit_data.py` and `audit_results.py`: data and metric consistency checks.
- `report_nautilus/`: report source, measured values, and figures.

The root-level `models.py`, `train.py`, and `evaluate.py` retain the original
baseline implementation. The final GPU experiment uses `nautilus/payload/`.

| Model | Validation FID-2048 | Validation IS | Test FID-2048 | Test IS |
| --- | ---: | ---: | ---: | ---: |
| Original DCGAN, width 32, 20 epochs | 112.81 | 3.212 | 111.84 | 3.213 |
| Wider DCGAN, width 64, 40 epochs | 67.62 | 4.402 | 67.58 | 4.402 |
| Resize + spectral normalization + hinge, width 64, 40 epochs | 75.77 | 4.713 | — | — |

## Generate samples

With the dependencies installed in Python 3.12:

    python generate.py --checkpoint results/nautilus-v1/plain/generator_040.pt --device cpu --samples 16 --columns 4 --output wider_dcgan.png

## Reproduce

The raw image archive is not redistributed. Download version 1 of
[steubk/wikiart](https://www.kaggle.com/datasets/steubk/wikiart), and run:

    python prepare_data.py --archive wikiart-v1.zip --output data/wikiart64_rerun
    python audit_data.py --data data/wikiart64_rerun
    cmp data/wikiart64/manifest.csv data/wikiart64_rerun/manifest.csv

    python nautilus/payload/train.py --data data/wikiart64_rerun --output results/rerun_plain --device cuda --variant plain --width 64 --epochs 40
    python nautilus/payload/train.py --data data/wikiart64_rerun --output results/rerun_resize --device cuda --variant resize-sn --width 64 --epochs 40
    python nautilus/payload/evaluate.py --data data/wikiart64_rerun --checkpoint results/rerun_plain/generator_040.pt --output results/rerun_plain_validation --split validation --device cuda

See nautilus/run.sh and nautilus/job.yaml for the pinned CUDA container and
the actual selection/test pipeline. The two new runs used PyTorch 2.4.1+cu121;
requirements.txt describes the local Python environment. The original model's
code, final generator, and run records are included for comparison.

The source now includes concise English comments. Executable tokens and Python
syntax trees were verified unchanged for the experiment code. `nautilus/inputs.sha256`
contains current source checksums for the preflight check; `nautilus/inputs.original.sha256`
preserves the checksums used by the completed experiment. Dataset and model
checksums are unchanged.

## Included evidence

- Report source and figures in report_nautilus/.
- All 40 epochs of training history for each new model, model configurations,
  and their final generators in results/nautilus-v1/.
- The validation and held-out test records, independent metric audits, and
  seeded sample grids. The resize/SN variant did not advance to testing.
- Source dataset manifest and data audit. Full dataset tensors, optimizer
  states, and feature arrays are excluded from this submission package.
