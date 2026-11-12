"""Independently recompute recorded metrics from saved features and verify provenance."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pytorch_fid.fid_score import calculate_frechet_distance


def fid(a, b):
    # Recompute feature means and covariances in double precision.
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(calculate_frechet_distance(a.mean(0), np.cov(a, rowvar=False),
                                           b.mean(0), np.cov(b, rowvar=False)))


def inception_score(probabilities):
    probs = probabilities.astype(np.float64)
    # IS is exp(mean KL(p(class|image) || p(class))) over all generated images.
    marginal = probs.mean(0)
    return float(np.exp(np.mean(np.sum(probs * (np.log(probs + 1e-16)
                                               - np.log(marginal + 1e-16)), axis=1))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evaluation", type=Path, default=Path("results/evaluation"))
    p.add_argument("--checkpoint", type=Path, default=Path("results/main/generator_020.pt"))
    args = p.parse_args()
    metrics = json.loads((args.evaluation / "metrics.json").read_text())
    if "latent_seed" in metrics:
        assert metrics["latent_seed"] not in {metrics["training_noise_seed"], metrics["training_noise_seed"] + 1}
    # Reject metrics attributed to a different model checkpoint.
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == metrics["checkpoint_sha256"]
    features = np.load(args.evaluation / "features.npz", allow_pickle=False)
    for key in ["real", "generated", "real_probabilities", "generated_probabilities"]:
        assert len(features[key]) == metrics["samples"]
        assert np.isfinite(features[key]).all()
    for key in ["real_probabilities", "generated_probabilities"]:
        np.testing.assert_allclose(features[key].sum(1), 1, atol=1e-5)
    assert len(set(features["real_indices"].tolist())) == metrics["samples"]
    computed = dict(fid_2048=fid(features["generated"], features["real"]),
                    fid_1000=fid(features["generated_probabilities"], features["real_probabilities"]),
                    is_generated=inception_score(features["generated_probabilities"]),
                    is_real=inception_score(features["real_probabilities"]))
    # Allow numerical rounding differences, not changes to the recorded scores.
    for key, value in computed.items():
        np.testing.assert_allclose(value, metrics[key], rtol=1e-5, atol=1e-5, err_msg=key)
    output = {"passed": True, "independently_recomputed": computed,
              "checkpoint_hash_matches": True, "sample_count_matches": True,
              "real_indices_unique": True,
              "separate_evaluation_latent_seed": "latent_seed" in metrics}
    (args.evaluation / "audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
