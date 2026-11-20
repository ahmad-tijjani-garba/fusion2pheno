#!/bin/bash
#SBATCH --job-name=f2p_rebuild
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# Partitions, GPUs and resources are those of the cluster used for the paper; adapt them to yours.
# Rebuild FusionPath's labelled dataset with its own scripts (STAR-Fusion step skipped).
set -euo pipefail
P=${F2P_ROOT:?set F2P_ROOT}
export DATA_DIR=$P/data/fusionpath_rebuild
export PYTHONPATH=$P/external/fusionpath
S=$P/external/fusionpath/fusionpath/preprocess
M=$DATA_DIR/01_source_data/merged
source ${F2P_VENV:-${F2P_ROOT}/.venv}/bin/activate
cd "$P"

echo "== 02_0 ChimerDB combine";  python $S/01_datasets/02_0_preprocess_chimerdb.py | tail -8
echo "== 02_1 ChimerDB filter";   python $S/01_datasets/02_1_filter_chimerdb.py | grep -E "Original|After|removed"
echo "== 02_3 combine";           python $S/01_datasets/02_3_combine_fusionpdb_chimerdb.py
echo "== 02_4 dedup";             python $S/01_datasets/02_4_remove_duplicates.py $M/merged_fusions.csv $M/merged_fusions.filtered.csv | tail -3
echo "== 02_5 liftover";          python $S/01_datasets/02_5_liftover_to_hg38.py $M/merged_fusions.filtered.csv $P/external/fusionpath/fusionpath/data/hg19ToHg38.over.chain.gz $M/merged_fusions.filtered.hg38.csv | tail -3
echo "== 04 format (no STAR-Fusion)"; python $S/01_datasets/04_convert_format.py $M/merged_fusions.filtered.hg38.csv $M/merged_fusions.filtered.hg38.formatted.csv

echo "== 01 label (as published, incl. Non-Cancer case bug)"
python $S/02_labels/01_label_fusions.py --data_dir "$DATA_DIR/" \
  --input_file $M/merged_fusions.filtered.hg38.formatted.csv \
  --output_file $M/merged_fusions.filtered.hg38.formatted.labeled.csv | tail -8

echo "== 01 label (bug fixed: Non-Cancer -> NORMAL before labelling)"
python - "$M" <<'EOF'
import sys, pandas as pd
m = sys.argv[1]
d = pd.read_csv(f"{m}/merged_fusions.filtered.hg38.formatted.csv", low_memory=False)
n = (d["cancer"].astype(str).str.upper() == "NON-CANCER").sum()
d.loc[d["cancer"].astype(str).str.upper() == "NON-CANCER", "cancer"] = "NORMAL"
d.to_csv(f"{m}/merged_fusions.filtered.hg38.formatted.noncancerfix.csv", index=False)
print(f"recoded {n} Non-Cancer rows to NORMAL")
EOF
python $S/02_labels/01_label_fusions.py --data_dir "$DATA_DIR/" \
  --input_file $M/merged_fusions.filtered.hg38.formatted.noncancerfix.csv \
  --output_file $M/merged_fusions.filtered.hg38.formatted.noncancerfix.labeled.csv | tail -8

ls -l $M
