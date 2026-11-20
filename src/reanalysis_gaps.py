"""Gene-level bootstrap for the FusionPath rebuild and the domain-retention baseline (GPU-cluster runs), paired with the
existing models on the same splits. Writes results/reanalysis/{fpr,v2_domains}.{csv,md}."""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd

P = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reanalysis as RA  # noqa: E402

R = P / "results"

# ---------------------------------------------------------------- FusionPath-style splits
new = pd.read_csv(R / "fp_rev_fpr/predictions_FPR.csv.gz")
old = pd.concat([pd.read_csv(f) for f in sorted((R / "fp_rev").glob("predictions_*.csv.gz"))])
old = old[(old.labelset == "as_published") & old.model.isin(
    ["gene_name", "kmer3", "homology", "esm2_650m/mlp", "fuson_plm_native/mlp", "fusionpath_like/mlp"])]
q = pd.concat([old, new])
genes = sorted(set(q.hgene) | set(q.tgene))
models = sorted(q.model.unique())
wide = q.pivot_table(index=["split", "row_id", "hgene", "tgene", "label"], columns="model", values="score").reset_index()
comps = [("fusionpath_rebuild/full", "fusionpath_like/mlp"), ("fusionpath_rebuild/full", "gene_name"),
         ("fusionpath_rebuild/full", "fuson_plm_native/mlp"), ("fusionpath_rebuild/full", "esm2_650m/mlp"),
         ("fusionpath_rebuild/seq_dom", "fusionpath_like/mlp"), ("fusionpath_rebuild/full", "fusionpath_rebuild/seq_dom"),
         ("fusionpath_rebuild/dom_go", "gene_name"), ("fusionpath_rebuild/dom_go", "esm2_650m/mlp"),
         ("domain_retention", "gene_name"), ("domain_retention", "esm2_650m/mlp"), ("domain_retention", "kmer3")]
groups = {"random_row": ["random_row"], "random_seq": ["random_seq"], "pair_grouped": ["pair_grouped"],
          "genes_held_out": [f"genes_held_out_{k}" for k in range(1, 11)]}
rows = []
for g, sps in groups.items():
    blocks = [wide[wide.split == sp].dropna(subset=models) for sp in sps if (wide.split == sp).any()]
    blocks = [b for b in blocks if 0 < b.label.sum() < len(b)]
    if not blocks:
        continue
    pt, dr, per = RA.gene_boot(blocks, models, genes)
    rows += RA.summarise(pt, dr, per, models, comps, {"dataset": "fp_rebuild", "split": g})
    print("fp", g, len(blocks), flush=True)
fpr = pd.DataFrame(rows)
fpr.to_csv(R / "reanalysis/fpr.csv", index=False)
(R / "reanalysis/fpr.md").write_text("# FusionPath rebuild and domain retention (FusionPath-style labels)\n\n" + fpr.round(4).to_markdown(index=False) + "\n")

# ---------------------------------------------------------------- gene-role-free labels
import train_v2 as T  # noqa: E402
s, _ = T.load()
genes_of = dict(zip(s.pair, zip(s.hgene, s.tgene)))
genes2 = sorted(set(s.hgene) | set(s.tgene))
dom = pd.read_csv(R / "v2_rev/domains/predictions.csv.gz")
base = pd.concat([pd.read_csv(R / "v2_rev/main/predictions.csv.gz"), pd.read_csv(R / "v2_rev/matched/predictions.csv.gz")])
keep = ["shortcuts", "esm2_logreg", "esm2+shortcuts", "kmer3", "gene_name", "geometry", "length"]
rows = []
for sub, dq in dom.groupby("subset"):
    seeds = sorted(dq.seed.unique())
    bq = base[(base.subset == sub) & base.seed.isin(seeds) & base.model.isin(keep)]
    qq = pd.concat([dq, bq]).drop(columns=["both_unseen"], errors="ignore")
    qq = T.mark_both_unseen(qq, s)
    qq["hgene"] = qq.pair.map(lambda p: genes_of[p][0]); qq["tgene"] = qq.pair.map(lambda p: genes_of[p][1])
    ms = [m for m in ["domains", "domains+shortcuts"] + keep if m in set(qq.model)]
    cps = [("domains", "shortcuts"), ("domains", "esm2_logreg"), ("domains", "gene_name"), ("domains", "geometry"),
           ("domains+shortcuts", "shortcuts"), ("esm2_logreg", "domains")]
    for label, filt in [("all test pairs", False), ("both genes unseen", True)]:
        z = qq[qq.both_unseen.astype(bool)] if filt else qq
        w = z.pivot_table(index=["seed", "pair", "hgene", "tgene", "label"], columns="model", values="p").reset_index()
        blocks = [b.dropna(subset=ms) for _, b in w.groupby("seed")]
        blocks = [b for b in blocks if 0 < b.label.sum() < len(b)]
        if not blocks:
            continue
        pt, dr, per = RA.gene_boot(blocks, ms, genes2)
        rows += RA.summarise(pt, dr, per, ms, cps, {"dataset": "v2_domains", "subset": sub, "test_pairs": label,
                                                    "median_test_pairs": float(np.median([len(b) for b in blocks]))})
    print("v2", sub, flush=True)
v2d = pd.DataFrame(rows)
v2d.to_csv(R / "reanalysis/v2_domains.csv", index=False)
(R / "reanalysis/v2_domains.md").write_text("# Domain-retention baseline (gene-role-free labels)\n\n" + v2d.round(4).to_markdown(index=False) + "\n")
for t in (fpr, v2d):
    t = t.copy(); t["v"] = t.apply(lambda r: f"{r.AUROC:.3f} ({r.ci_lo:.3f}, {r.ci_hi:.3f})" + (f" {r.wins}" if isinstance(r.wins, str) else ""), axis=1)
    idx = ["name"]; col = "split" if "split" in t else "subset"
    if "test_pairs" in t:
        t = t[t.test_pairs == "all test pairs"]
    print(t.pivot_table(index=idx, columns=col, values="v", aggfunc="first").to_string())
