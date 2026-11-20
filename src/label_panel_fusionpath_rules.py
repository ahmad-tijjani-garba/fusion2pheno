"""Apply FusionPath's own labelling function to paediatric driver fusions.

Uses FusionPath's determine_label() unchanged (external/fusionpath), with
NCG (vogelstein_annotation), ClinGen and CIViC loaded the same way as in
FusionPath's load_data(). Fusions are treated as cancer-derived (FusionGDB/ChimerDB),
so the normal-tissue and STAR-Fusion benign rules do not fire.
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import sys
from pathlib import Path

import pandas as pd

ROOT = F2P_ROOT
sys.path.insert(0, str(ROOT / "external/fusionpath/fusionpath/preprocess/02_labels"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

fp = import_module("01_label_fusions")  # FusionPath's labelling module
from fusondb_paediatric_coverage import PANEL  # our 54-fusion paediatric panel

raw = ROOT / "data/raw"

# NCG roles: same parsing as FusionPath load_data()
ncg = pd.read_csv(raw / "ncg/NCG_cancerdrivers_annotation_supporting_evidence.tsv", sep="\t")
role = {}
for _, r in ncg.iterrows():
    sym, ann = str(r["symbol"]).strip().upper(), str(r["vogelstein_annotation"]).upper()
    if ann == "NAN":
        continue
    if "ONCOGENE" in ann:
        role[sym] = "ONCOGENE"
    elif "TSG" in ann or "TUMOR SUPPRESSOR" in ann:
        role[sym] = "TSG"

clingen = pd.read_csv(raw / "annotations/clingen_gene_validity.csv", skiprows=4)
clingen_genes = set(clingen["GENE SYMBOL"].dropna().str.upper())

civic = pd.read_csv(raw / "annotations/civic_nightly_FeatureSummaries.tsv", sep="\t")
civic_fus = civic[civic["feature_type"].str.lower() == "fusion"]["name"].dropna().astype(str)
civic_genes = set()
for item in civic_fus:
    for part in item.split("::"):
        civic_genes.add(part.strip().upper().replace("[", "").replace("]", ""))
(raw / "annotations/civic_fusions.csv").write_text("\n".join(civic_fus) + "\n")

print(f"NCG Vogelstein roles: {len(role)} genes; ClinGen genes: {len(clingen_genes)}; "
      f"CIViC fusion features: {len(civic_fus)} ({len(civic_genes)} genes)")

rows = []
for h, t, disease in PANEL:
    row = {"hgene": h, "tgene": t, "cancer": "CANCER", "primary_source": "FusionGDB"}
    label, reason = fp.determine_label(row, role, clingen_genes, civic_genes, {}, False, 0.6, 0.6)
    rows.append({
        "fusion": f"{h}::{t}", "disease": disease, "fusionpath_label": label, "reason": reason,
        "h_role": role.get(h, ""), "t_role": role.get(t, ""),
        "h_clingen": h in clingen_genes, "t_clingen": t in clingen_genes,
        "h_civic": h in civic_genes, "t_civic": t in civic_genes,
    })
out = pd.DataFrame(rows)
out.to_csv(ROOT / "results/panel_fusionpath_labels.csv", index=False)
print(out["fusionpath_label"].value_counts().to_string())
print()
print(out[["fusion", "disease", "fusionpath_label", "h_role", "t_role"]].to_string(index=False))
