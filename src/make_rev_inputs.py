"""Revision inputs: panel isoforms with the fixed parser (FASTA + table) and MANE wild-type proteins (FASTA)."""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import hashlib
import sys
from pathlib import Path

import pandas as pd

P = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_panel import panel_sequences  # noqa: E402

D = P / "derived"
D.mkdir(exist_ok=True)
full = pd.read_csv(P / "data/fusionpath_rebuild/01_source_data/merged/merged_fusions.filtered.hg38.formatted.labeled.csv",
                   low_memory=False, usecols=["hgene", "tgene", "aa_seq", "cancer_label"])
full["hgene"] = full.hgene.fillna("?").astype(str).str.upper()
full["tgene"] = full.tgene.fillna("?").astype(str).str.upper()
ps = panel_sequences(full)
ps.drop(columns=["aa_seq"]).to_csv(D / "panel_isoforms.csv", index=False)
with open(D / "panel_isoforms.fasta", "w") as f:
    for sh, s in ps.drop_duplicates("sha1")[["sha1", "aa_seq"]].itertuples(index=False):
        f.write(f">{sh}\n{s}\n")
print(f"panel: {ps.fusion.nunique()} fusions, {ps.sha1.nunique()} unique isoforms")

wt = pd.read_csv(P / "external/fusionpath/fusionpath/data/gene_wt_mane_sequences.csv")
print("MANE columns:", list(wt.columns)[:8], len(wt))
seqcol = [c for c in wt.columns if "seq" in c.lower()][0]
wt = wt.dropna(subset=[seqcol])
with open(D / "wt_mane.fasta", "w") as f:
    for s in sorted(set(wt[seqcol].astype(str).str.replace("*", "", regex=False).str.upper())):
        f.write(f">{hashlib.sha1(s.encode()).hexdigest()}\n{s}\n")
print("wild-type proteins:", wt[seqcol].nunique())
