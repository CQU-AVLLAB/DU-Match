#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
dataset=${1:-pascal}
split=${2:-366}
gpus=${3:-1}
port=${4:-29501}
save_path=${5:-exp/$dataset/du_match/dinov2_base/$split}
python_path=${DUMATCH_PYTHON:-/home/chenyang/miniconda3/envs/unimatchv2/bin/python}
config="configs/${dataset}_du_match.yaml"
test -f "$config" || { echo "Missing config: $config" >&2; exit 1; }
mkdir -p "$save_path"
"$python_path" -m torch.distributed.run --nproc_per_node="$gpus" --master_addr=localhost --master_port="$port" \
    du_match.py --config "$config" --labeled-id-path "splits/$dataset/$split/labeled.txt" \
    --unlabeled-id-path "splits/$dataset/$split/unlabeled.txt" --save-path "$save_path" \
    2>&1 | tee -a "$save_path/out.log"