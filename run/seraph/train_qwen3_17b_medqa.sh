#!/usr/bin/env bash
#SBATCH --job-name=lora17_medqa
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=29G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
: "${SLURM_JOB_ID:?submit through Slurm}"

lora_project="$PWD/baseline/lora_exp"
results_root="$PWD/results/lora_17b_medqa"
run_dir="$results_root/job_$SLURM_JOB_ID"
adapter_root="/data/$USER/models/lora_exp/phase3c_qwen3_17b_medqa_seed2026/job_$SLURM_JOB_ID"
mkdir -p -- "$results_root"
mkdir -- "$run_dir"
status=FAILED
trap 'printf "status=%s\nexit_code=%s\njob_id=%s\n" "$status" "$?" "$SLURM_JOB_ID" > "$run_dir/status.txt"' EXIT

git rev-parse HEAD > "$run_dir/commit.txt"
git status --short > "$run_dir/git_status.txt"
printf 'Results: %s\n' "$run_dir"

conda_base=$(conda info --base)
source "$conda_base/etc/profile.d/conda.sh"
conda activate lab
set -u
export TOKENIZERS_PARALLELISM=false

python - "$lora_project" "$run_dir" "$adapter_root" <<'PY'
import json
import os
from pathlib import Path
import sys

import torch
import yaml

project, run, output = map(Path, sys.argv[1:])
pair_path = project.parent / "crosskv/configs/figure2_qwen3_1p7b_4b.json"
pair = json.loads(pair_path.read_text())
if pair["source_model"] != "Qwen/Qwen3-1.7B":
    raise RuntimeError(f"Unexpected source model in {pair_path}")

cfg = yaml.safe_load((project / "configs/phase3c_cross_domain.yaml").read_text())["phase3c"]
dataset = Path(os.path.expandvars(cfg["dataset_root"]))
for path in (dataset / "manifest.json", dataset / "medqa/train.parquet", dataset / "medqa/validation.parquet"):
    if not path.is_file():
        raise FileNotFoundError(f"Phase 3C data missing: {path}")
if json.loads((dataset / "manifest.json").read_text()).get("status") != "PASS":
    raise RuntimeError(f"Phase 3C data manifest is not PASS: {dataset}")

if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported(including_emulation=False):
    raise RuntimeError("Allocate one GPU with native BF16 support.")
print(f"GPU: {torch.cuda.get_device_name(0)}; torch={torch.__version__}; CUDA={torch.version.cuda}")

model = yaml.safe_load((project / cfg["model_config"]).read_text())["model"]
model.update(model_id=pair["source_model"], revision=pair["source_revision"])
model_path = run / "model.yaml"
model_path.write_text(yaml.safe_dump({"model": model}, sort_keys=False))
cfg["model_config"] = str(model_path)
cfg["adapter_output_root"] = str(output)
cfg["datasets"] = {"medqa": cfg["datasets"]["medqa"]}
(run / "train.yaml").write_text(yaml.safe_dump({"phase3c": cfg}, sort_keys=False))
print(f"Training {model['model_id']} at {model['revision']} on MedQA")
PY

python -u "$lora_project/scripts/12_train_cross_domain.py" --config "$run_dir/train.yaml"

python - "$run_dir" "$adapter_root" <<'PY'
import json
from math import isfinite
from pathlib import Path
import sys

from safetensors import safe_open
import yaml

run, output = map(Path, sys.argv[1:])
summary = json.loads((output / "train_summary.json").read_text())
record = summary["summaries"]["medqa"]
if summary.get("status") != "PASS" or record.get("status") != "PASS":
    raise RuntimeError("MedQA training did not finish with PASS.")
if not isfinite(float(record["eval_metrics"]["eval_loss"])):
    raise RuntimeError("MedQA validation loss is not finite.")

adapter_dir = Path(record["adapter_dir"]).resolve()
if not adapter_dir.is_relative_to(output.resolve()):
    raise RuntimeError(f"Adapter was saved outside this job's output: {adapter_dir}")
model = yaml.safe_load((run / "model.yaml").read_text())["model"]
adapter = json.loads((adapter_dir / "adapter_config.json").read_text())
if (adapter.get("peft_type") != "LORA"
        or adapter.get("base_model_name_or_path") != model["model_id"]
        or adapter.get("revision") not in (None, model["revision"])):
    raise RuntimeError(f"Unexpected adapter identity: {adapter_dir}")
with safe_open(adapter_dir / "adapter_model.safetensors", framework="numpy", device="cpu") as weights:
    if not weights.keys():
        raise RuntimeError("Adapter weights are empty.")

(run / "adapter_dir.txt").write_text(f"{adapter_dir}\n")
print(f"MedQA adapter ready: {adapter_dir}")
PY
status=COMPLETE
