"""Domain-retention baseline on the gene-role-free labels (reviewer M7), same design as train_v2.py / train_v2_matched.py.

Models (logistic regression, C chosen on validation pair AUROC, pair weights as train_v2):
  domains            11 Pfam retention features (partner wild-type domains kept by the fusion, kinase domain, ...)
  domains+shortcuts  the same plus the shortcuts (length, genomic layout, role, gene names)
Subsets: AB_all and AB_nonRT (seeds 42, 1-10); single sources (seeds 42, 1-20), as in the main analysis.
Inputs: data/domains/v2_domain_features.csv.gz (export_domain_features.py). Output: results/v2_rev/domains/predictions.csv.gz
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import train_v2 as T
import train_v2_checks as C
import train_v2_matched as MT

OUT = T.P / "results/v2_rev/domains"
OUT.mkdir(parents=True, exist_ok=True)
s, _ = T.load()
s = s.join(C.provenance(s, s), on=["pair", "sha1"])
F = pd.read_csv(T.P / "data/domains/v2_domain_features.csv.gz")
cols = [c for c in F.columns if c not in ("seq_sha1", "hgene", "tgene")]
F = F.drop_duplicates(["seq_sha1", "hgene", "tgene"]).set_index(["seq_sha1", "hgene", "tgene"])
Dm = F.reindex(pd.MultiIndex.from_arrays([s.sha1, s.hgene, s.tgene]))[cols].fillna(0).to_numpy(np.float32)
print(f"{len(s):,} rows; domain features for {np.mean(Dm[:, cols.index('fusion_domains')] > 0):.1%} with a Pfam domain", flush=True)
pos = {i: k for k, i in enumerate(s.index)}
allpairs = s.drop_duplicates("pair")[["pair", "hgene", "tgene"]]
plan = [(sub, T.SEEDS) for sub in ("AB_all", "AB_nonRT")] + \
       [(sub, [42, *range(1, 21)]) for sub in ("A_fusionpath", "fusiongdb2", "fusionpath_db", "fuson_db")]
preds = []
t0 = time.time()
for sub, seeds in plan:
    if sub in ("AB_all", "AB_nonRT"):
        d0 = s if sub == "AB_all" else s[s.rt == 0]
    else:
        d0 = MT.subset(s, sub)
    for seed in seeds:
        part, _ = T.split(allpairs, seed)
        d = d0.assign(part=d0.pair.map(part))
        test_seqs = set(d.sha1[d.part == "test"])
        tr = d[(d.part == "train") & ~d.sha1.isin(test_seqs)]
        va = d[(d.part == "val") & ~d.sha1.isin(test_seqs)]
        te = d[d.part == "test"]
        n = [(x.drop_duplicates("pair").label == c).sum() for x in (tr, va, te) for c in (0, 1)]
        if min(n) < 2:
            continue
        vocab = {g: j for j, g in enumerate(sorted(set(tr.hgene) | set(tr.tgene)))}
        ytr, wtr = tr.label.to_numpy(), T.weights(tr)
        dm = lambda x: Dm[[pos[i] for i in x.index]]  # noqa: E731
        for name, f in [("domains", dm), ("domains+shortcuts", lambda x: np.hstack([dm(x), T.features("shortcuts", x, vocab, None)]))]:
            Fs = [f(x) for x in (tr, va, te)]
            p, Cc = T.fit_logreg(Fs[0], ytr, wtr, Fs[1], va, Fs[2])
            preds.append(T.pair_scores(te, p).assign(subset=sub, seed=seed, model=name, C=Cc))
        g = preds[-2]
        print(f"{sub} seed {seed}: test {te.pair.nunique()} pairs, domains pair AUROC {roc_auc_score(g.label, g.p):.3f}, "
              f"{time.time() - t0:.0f}s", flush=True)
pr = T.mark_both_unseen(pd.concat(preds).reset_index(), s)
pr.to_csv(OUT / "predictions.csv.gz", index=False)
ps = T.per_seed(pr)
ps.to_csv(OUT / "metrics_per_seed.csv", index=False)
(OUT / "summary.md").write_text("# Domain-retention baseline (gene-role-free labels)\n\n" +
                                ps.groupby(["subset", "model"]).auroc.agg(["mean", "std", "count"]).round(3).to_markdown() + "\n")
print((OUT / "summary.md").read_text())
