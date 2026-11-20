"""Gene-level bootstrap for the ESM-2 650M probes run on a GPU cluster (attention pooling, LoRA), paired with the frozen
models and baselines on the same gene assignments. Writes results/reanalysis/probes.{csv,md}."""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd

P = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reanalysis as RA  # noqa: E402
import train_v2 as T  # noqa: E402

R = P / "results"
s, _ = T.load()
genes_of = dict(zip(s.pair, zip(s.hgene, s.tgene)))
genes = sorted(set(s.hgene) | set(s.tgene))
probe = pd.concat([pd.read_csv(f) for f in sorted((R / "v2_rev/probes").glob("predictions_*.csv.gz"))], ignore_index=True)
base = pd.concat([pd.read_csv(R / "v2_rev/main/predictions.csv.gz"), pd.read_csv(R / "v2_rev/matched/predictions.csv.gz")],
                 ignore_index=True)
keep = ["shortcuts", "source", "shortcuts+source", "esm2_logreg", "esm2_mlp", "esm2+shortcuts", "fuson_logreg"]
rows = []
for sub, pq in probe.groupby("subset"):
    for pm, pg in pq.groupby("model"):
        seeds = sorted(pg.seed.unique())
        bq = base[(base.subset == sub) & base.seed.isin(seeds) & base.model.isin(keep)]
        q = pd.concat([pg, bq], ignore_index=True).drop(columns=["both_unseen"], errors="ignore")
        q = T.mark_both_unseen(q, s)
        q["hgene"] = q.pair.map(lambda p: genes_of[p][0]); q["tgene"] = q.pair.map(lambda p: genes_of[p][1])
        models = [m for m in [pm] + keep if m in set(q.model)]
        comps = [(pm, "shortcuts"), (pm, "esm2_logreg"), (pm, "esm2_mlp"), (pm, "shortcuts+source"), (pm, "source")]
        for label, filt in [("all test pairs", False), ("both genes unseen", True)]:
            qq = q[q.both_unseen.astype(bool)] if filt else q
            wide = qq.pivot_table(index=["seed", "pair", "hgene", "tgene", "label"], columns="model", values="p").reset_index()
            blocks = [g.dropna(subset=models) for _, g in wide.groupby("seed")]
            blocks = [b for b in blocks if 0 < b.label.sum() < len(b)]
            if not blocks:
                continue
            pt, dr, per = RA.gene_boot(blocks, models, genes)
            rows += RA.summarise(pt, dr, per, models, comps,
                                 {"dataset": "probes", "subset": sub, "probe": pm, "test_pairs": label,
                                  "median_test_pairs": float(np.median([len(b) for b in blocks]))})
        print(f"{sub}/{pm}: {len(seeds)} assignments", flush=True)
out = pd.DataFrame(rows)
out.to_csv(R / "reanalysis/probes.csv", index=False)
(R / "reanalysis/probes.md").write_text("# ESM-2 650M probes (GPU cluster): gene-resampled 95% CIs\n\n" + out.round(4).to_markdown(index=False) + "\n")
print(out[out.kind == "difference"].round(3).to_string(index=False))
