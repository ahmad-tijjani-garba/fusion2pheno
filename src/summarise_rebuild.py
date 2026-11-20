"""Summarise the rebuilt FusionPath dataset and where paediatric drivers end up."""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import sys
from pathlib import Path

import pandas as pd

P = F2P_ROOT
M = P / "data/fusionpath_rebuild/01_source_data/merged"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusondb_paediatric_coverage import PANEL, names  # noqa: E402

out = []
for tag, fn in [("as_published", "merged_fusions.filtered.hg38.formatted.labeled.csv"),
                ("noncancer_fixed", "merged_fusions.filtered.hg38.formatted.noncancerfix.labeled.csv")]:
    d = pd.read_csv(M / fn, low_memory=False, usecols=["hgene", "tgene", "cancer", "primary_source",
                                                        "aa_seq", "cancer_label", "label_reason"])
    d["hgene"] = d["hgene"].astype(str).str.upper()
    d["tgene"] = d["tgene"].astype(str).str.upper()
    lab = d[d["cancer_label"].isin(["Benign", "Pathogenic"])]
    out += [f"## {tag}", "",
            f"- Records: {len(d):,}; labelled (Benign+Pathogenic): {len(lab):,}; "
            f"Uncertain (dropped): {(d.cancer_label == 'Uncertain').sum():,}",
            f"- Label counts: {d.cancer_label.value_counts().to_dict()}",
            f"- Pathogenic share of labelled set: {(lab.cancer_label == 'Pathogenic').mean():.1%}",
            f"- By source: {pd.crosstab(d.primary_source, d.cancer_label).to_dict('index')}",
            f"- Top label reasons: {d.label_reason.value_counts().head(6).to_dict()}",
            f"- Unique head::tail pairs in labelled set: {lab[['hgene','tgene']].drop_duplicates().shape[0]:,}; "
            f"unique genes: {len(set(lab.hgene) | set(lab.tgene)):,}",
            ""]
    if tag == "noncancer_fixed":
        rows = []
        for h, t, disease in PANEL:
            sub = d[d.hgene.isin(names(h)) & d.tgene.isin(names(t))]
            vc = sub.cancer_label.value_counts().to_dict()
            rows.append({"fusion": f"{h}::{t}", "disease": disease, "records": len(sub),
                         "Pathogenic": vc.get("Pathogenic", 0), "Benign": vc.get("Benign", 0),
                         "Uncertain": vc.get("Uncertain", 0)})
        pan = pd.DataFrame(rows)
        pan.to_csv(P / "results/rebuild_panel_records.csv", index=False)
        present = pan[pan.records > 0]
        out += ["## Paediatric panel inside the rebuilt dataset (Non-Cancer fix applied)", "",
                f"- Panel fusions present: {len(present)}/{len(pan)}; records: {pan.records.sum():,}",
                f"- Records kept as Pathogenic: {pan.Pathogenic.sum():,}; Benign: {pan.Benign.sum():,}; "
                f"**dropped as Uncertain: {pan.Uncertain.sum():,}**",
                "", present.to_markdown(index=False), ""]

text = "\n".join(out)
(P / "results/rebuild_summary.md").write_text("# FusionPath dataset rebuild: summary\n\n" + text + "\n")
print(text)
