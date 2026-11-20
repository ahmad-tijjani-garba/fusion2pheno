#!/bin/bash
#SBATCH --job-name=f2p_score_panel
#SBATCH --partition=gpul40,gpua4000,gpu4070ti,gpu3060,gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
# Partitions, GPUs and resources are those of the cluster used for the paper; adapt them to yours.
# retrain classifiers (panel records excluded) and score paediatric panel fusions.
set -euo pipefail
P=${F2P_ROOT:?set F2P_ROOT}
export HF_HOME=$P/models/hf_cache HF_HUB_OFFLINE=1
cd "$P"; source ${F2P_VENV:-${F2P_ROOT}/.venv}/bin/activate
python scripts/score_panel.py 2>&1 | grep --line-buffered -v -E "UNEXPECTED|MISSING|Loading weights|^Key|^---|^Notes|can be ignored|newly initialized|LOAD REPORT|^$|ConvergenceWarning|warnings.warn"
