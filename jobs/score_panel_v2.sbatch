#!/bin/bash
#SBATCH --job-name=f2p_panel_v2
#SBATCH --partition=gpul40,gpua4000,gpu4070ti,gpu3060,gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# Partitions, GPUs and resources are those of the cluster used for the paper; adapt them to yours.
# embed the paediatric panel isoforms (inference only), train small
# classifiers on the strict v2 labels (5 calibration repeats) and score the panel. No pLM fine-tuning.
set -euo pipefail
cd ${F2P_ROOT:?set F2P_ROOT}/src
source ${F2P_VENV:-${F2P_ROOT}/.venv}/bin/activate
export HF_HOME=${F2P_ROOT:?set F2P_ROOT}/models/hf_cache HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u score_panel_v2.py 2>&1 | grep --line-buffered -v -E "Warning|warnings.warn|_methods|ret = "
