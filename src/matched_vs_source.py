"""Matched set: embeddings and embeddings + shortcuts + source compared with data-source features alone (saved
predictions of round3b.py; no training). Gene-resampled 95% CIs over the full gene set, as in the manuscript."""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd

P = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reanalysis as RA  # noqa: E402
import train_v2 as T  # noqa: E402

s, _ = T.load()
genes = sorted(set(s.hgene) | set(s.tgene))
pr = pd.read_csv(P / "results/round3/matched_predictions.csv.gz")
models = ["source", "shortcuts", "esm2_logreg", "fuson_logreg", "esm2+shortcuts+source", "fuson+shortcuts+source"]
comps = [("esm2_logreg", "source"), ("fuson_logreg", "source"), ("esm2+shortcuts+source", "source"),
         ("fuson+shortcuts+source", "source")]
rows = []
for label, both in (("all test pairs", False), ("both genes unseen", True)):
    q = pr[pr.both_unseen.astype(bool)] if both else pr
    wide = q.pivot_table(index=["seed", "pair", "hgene", "tgene", "label"], columns="model", values="p").reset_index()
    bl = [b.dropna(subset=models) for _, b in wide.groupby("seed")]
    bl = [b for b in bl if 0 < b.label.sum() < len(b)]
    pt, dr, per = RA.gene_boot(bl, models, genes)
    rows += RA.summarise(pt, dr, per, models, comps, {"test_pairs": label, "n_assignments": len(bl)})
out = pd.DataFrame(rows)
out.to_csv(P / "results/round3/matched_vs_source.csv", index=False)
print(out[out.kind == "difference"].round(3).to_string(index=False))
