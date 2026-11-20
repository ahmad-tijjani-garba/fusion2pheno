#!/bin/bash
#SBATCH --job-name=f2p_pooling
#SBATCH --partition=gpul40,gpua4000,gpu4070ti,gpu3060,gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
# Partitions, GPUs and resources are those of the cluster used for the paper; adapt them to yours.
# re-embed with partner/junction pooling (inference) and train/evaluate MLPs per pooling.
set -euo pipefail
P=${F2P_ROOT:?set F2P_ROOT}
export HF_HOME=$P/models/hf_cache HF_HUB_OFFLINE=1
cd "$P"; source ${F2P_VENV:-${F2P_ROOT}/.venv}/bin/activate
F="UNEXPECTED|MISSING|Loading weights|^Key|^---|^Notes|can be ignored|newly initialized|LOAD REPORT|^$|ConvergenceWarning|warnings.warn"
nvidia-smi --query-gpu=name --format=csv,noheader
python scripts/embed_pooled.py facebook/esm2_t33_650M_UR50D esm2_650m 2>&1 | grep --line-buffered -v -E "$F"
python scripts/embed_pooled.py ChatterjeeLab/FusOn-pLM fuson_plm 2>&1 | grep --line-buffered -v -E "$F"
python scripts/eval_pooling.py 2>&1 | grep --line-buffered -v -E "$F"
