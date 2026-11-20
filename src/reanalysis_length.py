"""Performance by protein length (reviewer M8): AUROC within length bins, gene-resampled 95% CIs, from saved predictions.

FusionPath-style labels: held-out genes (10 assignments), records binned by fusion-protein length.
Gene-role-free labels: held-out genes (11 assignments, all sources), pairs binned by the median length of their sequences.
Also the windowing check (FusOn-pLM native 2,000-residue windows minus 1,022-residue windows) within each bin.
Writes results/reanalysis/length.{csv,md}.
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd

P = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reanalysis as RA  # noqa: E402

R = P / "results"
rows = []

# ---------------------------------------------------------------- FusionPath-style labels
import fp_benchmark_rev as FB  # noqa: E402
d = FB.load("as_published")
seqlen = d.seq.str.len()
pr = pd.concat([pd.read_csv(f) for f in sorted((R / "fp_rev").glob("predictions_*.csv.gz"))])
pr = pr[(pr.labelset == "as_published") & pr.split.str.startswith("genes_held_out_")]
pr["length"] = pr.row_id.map(seqlen)
models = ["esm2_650m/mlp", "esm2_3b/mlp", "prot_t5/mlp", "fuson_plm/mlp", "fuson_plm_native/mlp", "fusionpath_like/mlp",
          "kmer3", "homology"]
wide = pr[pr.model.isin(models)].pivot_table(index=["split", "row_id", "hgene", "tgene", "label", "length"],
                                             columns="model", values="score").reset_index()
wide["protein_length"] = wide.length.astype(float)
models_l = models + ["protein_length"]
genes = sorted(set(pr.hgene) | set(pr.tgene))
bins = [(0, 500, "<=500"), (500, 1000, "501-1,000"), (1000, 2000, "1,001-2,000"), (2000, 10 ** 6, ">2,000")]
comps = [("fuson_plm_native/mlp", "fuson_plm/mlp"), ("fuson_plm/mlp", "esm2_650m/mlp"), ("esm2_650m/mlp", "kmer3")]
for lo, hi, lab in [(0, 10 ** 6, "all")] + bins:
    q = wide[(wide.length > lo) & (wide.length <= hi)]
    blocks = [g.dropna(subset=models) for _, g in q.groupby("split")]
    blocks = [b for b in blocks if b.label.sum() >= 3 and (1 - b.label).sum() >= 3]
    if not blocks:
        continue
    pt, dr, per = RA.gene_boot(blocks, models_l, genes)
    rows += RA.summarise(pt, dr, per, models_l, comps, {
        "dataset": "fusionpath_style", "length_bin": lab, "n_records": int(np.median([len(b) for b in blocks])),
        "positive_share": float(np.mean([b.label.mean() for b in blocks]))})
    print("fp", lab, len(blocks), "assignments", flush=True)

# ---------------------------------------------------------------- gene-role-free labels
import train_v2 as T  # noqa: E402
s, _ = T.load()
lencol = "length" if "length" in s.columns else None
plen = (s.groupby("pair")[lencol].median() if lencol else s.assign(L=s.seq.str.len()).groupby("pair").L.median())
genes_of = dict(zip(s.pair, zip(s.hgene, s.tgene)))
genes2 = sorted(set(s.hgene) | set(s.tgene))
v = pd.read_csv(R / "v2_rev/main/predictions.csv.gz")
v = v[v.subset == "AB_all"]
m2 = ["shortcuts", "esm2_logreg", "fuson_logreg", "fuson_native_logreg", "prot_t5_logreg", "esm2+shortcuts"]
w2 = v[v.model.isin(m2)].pivot_table(index=["seed", "pair", "label"], columns="model", values="p").reset_index()
w2["length"] = w2.pair.map(plen); w2["protein_length"] = w2.length.astype(float)
w2["hgene"] = w2.pair.map(lambda p: genes_of[p][0]); w2["tgene"] = w2.pair.map(lambda p: genes_of[p][1])
m2l = m2 + ["protein_length"]
comps2 = [("esm2+shortcuts", "shortcuts"), ("fuson_native_logreg", "fuson_logreg"), ("esm2_logreg", "shortcuts")]
for lo, hi, lab in [(0, 10 ** 6, "all"), (0, 500, "<=500"), (500, 1000, "501-1,000"), (1000, 10 ** 6, ">1,000")]:
    q = w2[(w2.length > lo) & (w2.length <= hi)]
    blocks = [g.dropna(subset=m2) for _, g in q.groupby("seed")]
    blocks = [b for b in blocks if b.label.sum() >= 3 and (1 - b.label).sum() >= 3]
    if not blocks:
        continue
    pt, dr, per = RA.gene_boot(blocks, m2l, genes2)
    rows += RA.summarise(pt, dr, per, m2l, comps2, {
        "dataset": "gene_role_free_all_sources", "length_bin": lab, "n_records": int(np.median([len(b) for b in blocks])),
        "positive_share": float(np.mean([b.label.mean() for b in blocks]))})
    print("v2", lab, len(blocks), "assignments", flush=True)

out = pd.DataFrame(rows)
out.to_csv(R / "reanalysis/length.csv", index=False)
(R / "reanalysis/length.md").write_text("# Performance by protein length (gene-resampled 95% CIs)\n\n" + out.round(4).to_markdown(index=False) + "\n")
print(out.round(3).to_string(index=False))
