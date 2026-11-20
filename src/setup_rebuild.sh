#!/bin/bash
# Set up DATA_DIR for the FusionPath rebuild and run the FusionPDB step.
set -euo pipefail
P=${F2P_ROOT:?set F2P_ROOT}
D=$P/data/fusionpath_rebuild
mkdir -p "$D"/01_source_data/{ChimerDB,FusionPDB,merged} "$D"/fusion_annotations

ln -sf "$P"/data/raw/chimerdb4/ChimerSeq4.xlsx "$D"/01_source_data/ChimerDB/ChimerSeq4.xlsx
ln -sf "$P"/data/raw/fusiongdb2/combinedFGDB2genes_ORF_analyzed_gencode_h19v19_real_Inframe_only_transcript_seq_with_orffinder_result.txt "$D"/01_source_data/FusionPDB/
ln -sf "$P"/data/raw/ncg/NCG_cancerdrivers_annotation_supporting_evidence.tsv "$D"/fusion_annotations/
ln -sf "$P"/data/raw/annotations/clingen_gene_validity.csv "$D"/fusion_annotations/Clingen-Gene-Disease-Summary-2026-01-05.csv
ln -sf "$P"/data/raw/annotations/civic_fusions.csv "$D"/fusion_annotations/civic_fusions.csv

cat > "$D"/README.md <<'EOF'
FusionPath dataset rebuild (DATA_DIR for external/fusionpath scripts).
Inputs are symlinks into data/raw/. Differences from the original:
- ClinGen file is the 2026-11-15 download, linked under FusionPath's expected name (theirs: 2026-01-05).
- CIViC fusions taken from the 2026-11-15 nightly FeatureSummaries (feature_type == Fusion).
- NCG from Zenodo 16024856 (NCG site offline).
- FusionGDB2 table from the Internet Archive (2025-05-25 snapshot of the 2022-03-24 file).
- STAR-Fusion normal-tissue step (27 ENA RNA-seq sets) not run.
EOF

gunzip -c "$P"/data/raw/chimerdb4/ChimerDB4.0_sequence.txt.gz > "$D"/01_source_data/ChimerDB/ChimerDB4.0_sequence.txt

source ${F2P_VENV:-${F2P_ROOT}/.venv}/bin/activate
F="$D"/01_source_data/FusionPDB/combinedFGDB2genes_ORF_analyzed_gencode_h19v19_real_Inframe_only_transcript_seq_with_orffinder_result
python "$P"/external/fusionpath/fusionpath/preprocess/01_datasets/02_2_preprocess_fusionpdb.py "$F.txt" "$F.filtered.csv"
echo "FusionPDB filtered rows (incl. header): $(wc -l < "$F.filtered.csv")"
echo "Top cancer types:"; cut -d, -f5 "$F.filtered.csv" | sort | uniq -c | sort -rn | head -8
echo "ChimerDB sequence lines: $(wc -l < "$D"/01_source_data/ChimerDB/ChimerDB4.0_sequence.txt)"
