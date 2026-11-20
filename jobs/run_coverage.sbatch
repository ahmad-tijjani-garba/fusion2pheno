#!/bin/bash
#SBATCH --job-name=f2p_coverage
#SBATCH --partition=poc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
# Partitions, GPUs and resources are those of the cluster used for the paper; adapt them to yours.

set -euo pipefail
cd ${F2P_ROOT:?set F2P_ROOT}
source ${F2P_VENV:-${F2P_ROOT}/.venv}/bin/activate
python scripts/fusondb_paediatric_coverage.py --csv data/raw/fuson_db/FusOn-DB.csv --out results
