#!/bin/bash
set -euo pipefail

# 当前脚本在 Image_Classification/src 里面
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_CONFIG="${PROJECT_ROOT}/configs/koalapViT.yaml"
TRAIN_PY="${PROJECT_ROOT}/src/train.py"

OUT_ROOT="${PROJECT_ROOT}/sweep_koalap_focalnet_cifar10"
mkdir -p "${OUT_ROOT}/configs" "${OUT_ROOT}/logs"

# ===== Hyperparameter grid =====
LR_LIST=(
  1e-4
  2e-4
  3e-4
)

WD_LIST=(
  0.0
  1e-2
)

echo "PROJECT_ROOT = ${PROJECT_ROOT}"
echo "BASE_CONFIG  = ${BASE_CONFIG}"
echo "TRAIN_PY     = ${TRAIN_PY}"

# 检查文件是否真的存在
ls -lh "${BASE_CONFIG}"
ls -lh "${TRAIN_PY}"

cd "${PROJECT_ROOT}"

for lr in "${LR_LIST[@]}"; do
  for wd in "${WD_LIST[@]}"; do

    tag="lr${lr}_wd${wd}"
    tag="${tag//./p}"
    tag="${tag//-/m}"

    cfg="${OUT_ROOT}/configs/${tag}.yaml"
    log="${OUT_ROOT}/logs/${tag}.log"

    echo "=================================================="
    echo "Running: init_lr=${lr}, weight_decay=${wd}"
    echo "Base config: ${BASE_CONFIG}"
    echo "Config:      ${cfg}"
    echo "Log:         ${log}"
    echo "=================================================="

    python3 - "$BASE_CONFIG" "$cfg" "$lr" "$wd" <<'PY'
import sys
import yaml
from pathlib import Path

base_config, out_config, lr, wd = sys.argv[1:]

with open(base_config, "r") as f:
    cfg = yaml.safe_load(f)

cfg["init_lr"] = float(lr)

if "optimizer_kwargs" not in cfg or cfg["optimizer_kwargs"] is None:
    cfg["optimizer_kwargs"] = {}

cfg["optimizer_kwargs"]["weight_decay"] = float(wd)

Path(out_config).parent.mkdir(parents=True, exist_ok=True)

with open(out_config, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

    python3 "${TRAIN_PY}" --config "${cfg}" 2>&1 | tee "${log}"

  done
done