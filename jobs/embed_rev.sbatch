#!/bin/bash
#SBATCH --partition=gpul40,gpua4000,gpu4070ti,gpu3060,gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# Partitions, GPUs and resources are those of the cluster used for the paper; adapt them to yours.
# revision embeddings,
# inference only. Args: <hf_model_id> <tag> <family> <chunk> <hf_home> [extra args]
set -euo pipefail
cd ${F2P_ROOT:?set F2P_ROOT}
source ${F2P_VENV:-${F2P_ROOT}/.venv}/bin/activate
export HF_HOME=$5 HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u scripts/embed_any.py "$1" "$2" --family "$3" --chunk "$4" "${@:6}"
