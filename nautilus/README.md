# GPU training environment

The recorded WikiArt experiments used one NVIDIA A100 80 GB with the pinned
PyTorch 2.4.1 / CUDA 12.1 container in `job.yaml`. Additional packages were
NumPy 1.26.4, SciPy 1.14.1, PyTorch-Ignite 0.5.3, and pytorch-fid 0.3.0.

## Experiment configuration

- DCGAN baseline: base width 32, 20 epochs, trained on MPS.
- DCGAN: base width 64, 40 epochs, BCE adversarial losses.
- Alternative: base width 64, 40 epochs, nearest-neighbor resize-convolution
  in the generator, spectral normalization on discriminator convolutions,
  and hinge losses.

Both GPU configurations use seed 285, batch size 128, and Adam learning rate
0.0002 with betas (0.5, 0.999). Validation uses the same 5,000 real images
(seed 286) and 5,000 latent samples (seed 1286). Configuration selection is
based on final-epoch validation FID-2048. Training histories, configurations,
metrics, and checkpoint hashes are stored in `results/nautilus-v1/`.

## Running on another GPU system

The commands in the [project README](../README.md) run without Kubernetes.
Use a CUDA-compatible PyTorch environment for `--device cuda`. CPU generation
with the included plain DCGAN checkpoints is also supported.

`job.yaml` and `run.sh` describe the original cluster execution pipeline.
They assume project files are staged under `/work` on a pre-existing PVC
named `hw2-wikiart-gan-v1`, including the prepared dataset, evaluator weights,
checksum manifest, and `nautilus/input-ready` marker. The manifest covers
artifacts such as the image cache and Inception weights that are not included
in this repository. Cluster-specific storage and GPU resource names must be
configured for the target environment before using the job.

The runner verifies input checksums, copies the dataset to node-local scratch,
trains both configurations, evaluates validation checkpoints, selects by FID,
and evaluates the selected new configuration on test data. Checkpoints and
results are written under `/work/results/nautilus-v1`.

`inputs.original.sha256` preserves the original experiment's checksums.
`inputs.sha256` reflects the experiment source after English comments were
added. Those comment edits did not change the model or training computations.
