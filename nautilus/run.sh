#!/bin/bash
set -euo pipefail
cd /work
test -f nautilus/input-ready
echo 'Copying WikiArt to node-local scratch and validating the copy'
mkdir -p /tmp/hw2/data
cp -r data/wikiart64 /tmp/hw2/data/
awk '$2 ~ /^data\/wikiart64\//' nautilus/inputs.sha256 | (cd /tmp/hw2 && sha256sum -c -)
awk '$2 !~ /^data\/wikiart64\//' nautilus/inputs.sha256 | sha256sum -c -
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
python -m pip install --disable-pip-version-check --no-cache-dir numpy==1.26.4 scipy==1.14.1 pytorch-ignite==0.5.3 pytorch-fid==0.3.0
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c 'import torch, torchvision; assert torch.cuda.is_available(); print(torch.__version__, torchvision.__version__, torch.version.cuda)'
root=/work/results/nautilus-v1
mkdir -p "$root"
data=/tmp/hw2/data/wikiart64
src=/work/nautilus/payload
train() {
  local variant="$1" epochs="$2"
  local dest="$root/$variant"
  if [ -f "$dest/completed.json" ] && python -c 'import json,sys; assert json.load(open(sys.argv[1]))["epochs"]==int(sys.argv[2])' "$dest/completed.json" "$epochs"; then
    return
  fi
  local extra=()
  if [ -f "$dest/last.pt" ]; then extra=(--resume); fi
  python "$src/train.py" --data "$data" --output "$dest" --device cuda --width 64 --variant "$variant" --epochs "$epochs" "${extra[@]}"
}
evaluate() {
  local name="$1" ckpt="$2" split="$3"
  local dest="$root/${name}_${split}"
  if [ ! -f "$dest/metrics.json" ]; then
    python "$src/evaluate.py" --data "$data" --checkpoint "$ckpt" --output "$dest" --split "$split" --samples 5000 --seed 286 --latent-seed 1286 --device cuda --torch-home /work/data/torch
  fi
  if [ ! -f "$dest/audit.json" ]; then
    python "$src/audit_results.py" --evaluation "$dest" --checkpoint "$ckpt"
  fi
}
# Start with a real one-epoch pilot. Resume the same run after the gate.
if [ ! -f "$root/resize-sn/last.pt" ]; then train resize-sn 1; fi
echo 'A100 pilot passed'
evaluate original /work/results/main/generator_020.pt validation
train plain 40
evaluate plain "$root/plain/generator_040.pt" validation
train resize-sn 40
evaluate resize-sn "$root/resize-sn/generator_040.pt" validation
winner="$(python - "$root" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
scores = {x: json.loads((root/f'{x}_validation/metrics.json').read_text())['fid_2048']
          for x in ('original', 'plain', 'resize-sn')}
winner = min(scores, key=scores.get)
(root/'selection.json').write_text(json.dumps(dict(validation_fid=scores, winner=winner), indent=2)+'\n')
print(winner)
PY
)"
if [ "$winner" != original ]; then
  evaluate "$winner" "$root/$winner/generator_040.pt" test
fi
cat "$root/selection.json"
touch "$root/completed"
