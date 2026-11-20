"""Source-matched evaluation on the strict label set v2.

The checks showed that where a sequence comes from separates the classes (AUROC 0.95). Here both classes are
drawn from one source at a time, so source can no longer stand in for the label:
  A_fusionpath : our GENCODE translations of FusionPath breakpoints only (cleanest: one pipeline, one source)
  fusiongdb2   : sequences present in FusionGDB2
  fusionpath_db: sequences present in the FusionPath database
  fuson_db     : sequences present in FusOn-DB (FusOn-pLM was pretrained on FusOn-DB; ESM-2 is the fair model here)
Same splits, weighting, model selection, pair-level scoring and gene-level bootstrap as train_v2.py /
train_v2_checks.py; all models (shortcuts, source, pLM logreg/MLP, pLM + shortcuts (+ source)).
Outputs: results/v2_strict/matched/{predictions.csv.gz, metrics_per_seed.csv, sizes.csv, gene_bootstrap.csv, summary.md}
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import train_v2 as T
import train_v2_checks as C

OUT = T.P / "results/v2_strict/matched"
SUBSETS = ["A_fusionpath", "fusiongdb2", "fusionpath_db", "fuson_db"]


def subset(s, name):
    if name == "A_fusionpath":
        return s[(s.A_pipeline == 1) & (s.bp_fusionpath == 1)]
    return s[s[name] == 1]


def run(s, emb, subsets, seeds, mlp=True):
    allpairs = s.drop_duplicates("pair")[["pair", "hgene", "tgene"]]
    preds, sizes = [], []
    for sub in subsets:
        d0 = subset(s, sub).copy()
        d0["log_nseq"] = np.log10(d0.groupby("pair").sha1.transform("nunique"))
        u = d0.drop_duplicates("pair")
        sizes.append({"subset": sub, "pos_pairs": int((u.label == 1).sum()), "neg_pairs": int((u.label == 0).sum()),
                      "pos_seqs": d0[d0.label == 1].sha1.nunique(), "neg_seqs": d0[d0.label == 0].sha1.nunique(),
                      "neg_pairs_readthrough": int(((u.label == 0) & (u.rt == 1)).sum())})
        print(sizes[-1], flush=True)
        for seed in seeds:
            t0 = time.time()
            part, _ = T.split(allpairs, seed)
            d = d0.assign(part=d0.pair.map(part))
            test_seqs = set(d.sha1[d.part == "test"])
            tr = d[(d.part == "train") & ~d.sha1.isin(test_seqs)]
            va = d[(d.part == "val") & ~d.sha1.isin(test_seqs)]
            te = d[d.part == "test"]
            n = [(x.drop_duplicates("pair").label == c).sum() for x in (tr, va, te) for c in (0, 1)]
            if min(n) < 2:
                print(f"{sub} seed {seed}: too few pairs of a class (train/val/test neg,pos = {n}), skipped", flush=True)
                continue
            vocab = {g: j for j, g in enumerate(sorted(set(tr.hgene) | set(tr.tgene)))}
            ytr, wtr = tr.label.to_numpy(), T.weights(tr)
            jobs = [(m, None, lambda x, k, m=m: T.features(m, x, vocab, None))
                    for m in ["length", "geometry", "role", "gene_name", "shortcuts"]]
            jobs += [(m, None, lambda x, k, m=m: C.feats(m, x, vocab, None)) for m in ["source", "shortcuts+source"]]
            for k in emb:
                jobs += [(f"{k}_logreg", k, lambda x, k: emb[k][x.index.to_numpy()]),
                         (f"{k}+shortcuts", k, lambda x, k: T.features("plm+shortcuts", x, vocab, emb[k][x.index.to_numpy()])),
                         (f"{k}+shortcuts+source", k, lambda x, k: C.feats("shortcuts+source", x, vocab, emb[k][x.index.to_numpy()]))]
            for name, k, f in jobs:
                F = [f(x, k) for x in (tr, va, te)]
                p, Cc = T.fit_logreg(F[0], ytr, wtr, F[1], va, F[2])
                preds.append(T.pair_scores(te, p).assign(subset=sub, seed=seed, model=name, C=Cc))
            if mlp:
                for k in emb:
                    ix = [x.index.to_numpy() for x in (tr, va, te)]
                    p = T.fit_mlp(emb[k][ix[0]], ytr, wtr, emb[k][ix[1]], va, emb[k][ix[2]], seed)
                    preds.append(T.pair_scores(te, p).assign(subset=sub, seed=seed, model=f"{k}_mlp", C=np.nan))
            print(f"{sub} seed {seed}: train {n[0]}/{n[1]} val {n[2]}/{n[3]} test {n[4]}/{n[5]} (neg/pos pairs), "
                  f"{time.time() - t0:.0f}s", flush=True)
    return pd.concat(preds).reset_index(), pd.DataFrame(sizes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=T.SEEDS)
    ap.add_argument("--subsets", nargs="*", default=SUBSETS)
    ap.add_argument("--no-mlp", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    s, emb = T.load()
    s = s.join(C.provenance(s, s), on=["pair", "sha1"])
    pr, sizes = run(s, emb, a.subsets, a.seeds, mlp=not a.no_mlp)
    pr.to_csv(out / "predictions.csv.gz", index=False)
    sizes.to_csv(out / "sizes.csv", index=False)
    ps = T.per_seed(pr)
    ps.to_csv(out / "metrics_per_seed.csv", index=False)
    fz = s.groupby("pair").fuson_db.max()
    subs = [x for x in a.subsets if x in set(pr.subset)]
    md, tab = C.report(T.mark_both_unseen(pr, s), s, set(fz.index[fz == 0]), subs)
    md = md.replace("# Strict label set v2: follow-up checks", "# Strict label set v2: source-matched evaluation")
    used = ps.groupby("subset").seed.nunique().rename("assignments used")
    md = md.replace("\n## ", "\n" + sizes.set_index("subset").join(used).to_markdown() + "\n\n## ", 1)
    tab.to_csv(out / "gene_bootstrap.csv", index=False)
    (out / "summary.md").write_text(md)
    print(md)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
