# WikiArt GAN comparison on Nautilus

This experiment uses one A100 in the `ecepxie` namespace and the original
64 × 64 WikiArt cache. It writes to a dedicated PVC and does not alter the
previous report, model or local evaluation results.

- Original: width 32, 20 epochs, existing trained model.
- Wider DCGAN: width 64, 40 epochs, original objective and architecture.
- Resize + SN: width 64, 40 epochs, nearest-neighbor upsampling and convolution
  in the generator, spectral normalization on discriminator convolutions, and
  hinge losses. The first epoch doubles as a real-data CUDA pilot.

Both new runs use seed 285, batch size 128 and Adam at 2e-4. Comparisons use
the final epoch of each predeclared run. Each validation evaluation uses the
same 5,000 real validation images (seed 286), 5,000 latent draws (seed 1286),
and the original pool3 FID and IS protocol. We select by validation pool3 FID.
The winner is tested once if it is a new model; the original test metrics are
already recorded locally. The two new models share a training budget;
differences relative to the original may reflect both capacity and budget.
This first experiment does not estimate seed-to-seed variability.

`storage.yaml` defines the 12 GiB PVC and a time-limited, CPU-only transfer Pod.
`inputs.sha256` lists the staged input files. The transfer Pod is removed after
the integrity check and script upload. `job.yaml` runs `run.sh` on one A100;
its deadline is four hours, with one retry and persistent epoch checkpoints.
Dependencies install in the container layer. The active cache is copied to
local scratch. Results are persisted in `/work/results/nautilus-v1` on the PVC.

After the readability update, `inputs.sha256` reflects the commented source.
`inputs.original.sha256` preserves the original experiment's checksums. The
comments do not change executable experiment code, dataset contents, or weights.

Read-only status checks:

```bash
kubectl --context nautilus -n ecepxie get job hw2-wikiart-a100-v1
kubectl --context nautilus -n ecepxie logs job/hw2-wikiart-a100-v1 --tail=30
```

After the job succeeds, run `bash nautilus/collect.sh` from the HW2 folder to
retrieve the validation/test metrics, image grids, loss histories and generator
weights. The full feature arrays and optimizer states remain on the PVC.

The initial full-cache preflight found one mismatched 8 MiB chunk. The bad
chunk was located with `check_chunks.py`, re-uploaded with `repair_chunks.py`,
flushed and verified. The current job verifies the complete copy on node-local
scratch before consuming it. The first failure log is retained locally.
