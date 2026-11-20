"""Robustness check for partner/junction pooling vs whole-protein mean.

A. Gene-disjoint benchmark: 10 gene assignments (seeds 1..10). For each, MLP (3 seeds) per
   embedding x pooling {mean, head+tail, head+tail+junction}. Paired difference vs mean,
   Wilcoxon signed-rank test, count of assignments where the new pooling wins.
B. Paediatric panel: 5 calibration repeats (different 85/15 pair splits, seeds 101..105).
   Per repeat: fusions called (majority of isoforms above 90%-specificity threshold),
   long-fusion calls. Per fusion: share of repeats in which it is called.
Outputs: results/pooling_repeats_{A,B,B_fusions}.csv, results/pooling_repeats.md
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

P = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pooling import feats, load_pools, mlp_scores  # noqa: E402
from fusondb_paediatric_coverage import PANEL, names  # noqa: E402
from train_seq_models import gene_disjoint_ids, load_rows, metrics  # noqa: E402

SUF = sys.argv[1] if len(sys.argv) > 1 else ""  # e.g. "_up" for MANE + UniProt junctions
TAGS = [t + SUF for t in ("esm2_650m", "fuson_plm")]
VARS = ["mean", "head+tail", "head+tail+junction"]


def part_a(d, pools):
    rows = []
    for seed in range(1, 11):
        tr, va, te = gene_disjoint_ids(d, seed=seed)
        y = [d.loc[i, "label"].to_numpy() for i in (tr, va, te)]
        for tag in TAGS:
            pl, lut, _ = pools[tag]
            for v in VARS:
                X = [feats(pl, lut, d.loc[i, "sha1"], v) for i in (tr, va, te)]
                au, ap = metrics(y[2], mlp_scores(X[0], y[0], X[1], y[1], X[2]))
                rows.append({"seed": seed, "embedding": tag, "pooling": v, "n_test": len(te),
                             "pos": int(y[2].sum()), "AUROC": au, "AUPRC": ap})
        print(f"  A seed {seed} done", flush=True)
    return pd.DataFrame(rows)


def part_b(d, pools):
    panel_pairs = {(hh, tt) for h, t, _ in PANEL for hh in names(h) for tt in names(t)}
    d = d[[(h, t) not in panel_pairs for h, t in zip(d.hgene, d.tgene)]]
    # revision: first-run 46-fusion isoform list; pooled embeddings exist only for these isoforms
    prev = pd.read_csv(F2P_ROOT / "results/panel_scores.csv")[["fusion", "sha1"]]
    pairs = (d.hgene + "::" + d.tgene)
    upairs = pairs.unique()
    rows, frows = [], []
    for rep in range(101, 106):
        rng = np.random.default_rng(rep)
        cal_pairs = set(rng.choice(upairs, size=int(0.15 * len(upairs)), replace=False))
        in_cal = pairs.isin(cal_pairs).to_numpy()
        tr, cal = d[~in_cal], d[in_cal]
        ytr, ycal = tr.label.to_numpy(), cal.label.to_numpy()
        for tag in TAGS:
            pl, lut, idx = pools[tag]
            length = prev.sha1.map(dict(zip(idx.seq_sha1, idx.length)))
            for v in VARS:
                Xtr, Xcal, Xp = (feats(pl, lut, s, v) for s in (tr.sha1, cal.sha1, prev.sha1))
                both = mlp_scores(Xtr, ytr, Xcal, ycal, np.vstack([Xcal, Xp]))
                cs, ps = both[:len(Xcal)], both[len(Xcal):]
                thr = np.quantile(cs[ycal == 0], 0.90)
                g = prev.assign(call=ps >= thr, length=length).groupby("fusion")
                share = g.call.mean()
                long = g.length.median() > 1500
                called = share >= 0.5
                rows.append({"repeat": rep, "embedding": tag, "pooling": v, "cal_AUROC": roc_auc_score(ycal, cs),
                             "called": int(called.sum()), "long_called": int((called & long).sum()),
                             "short_called": int((called & ~long).sum()), "n": len(share), "long_n": int(long.sum())})
                for f in share.index:
                    frows.append({"repeat": rep, "embedding": tag, "pooling": v, "fusion": f,
                                  "long": bool(long[f]), "called_share": share[f], "called": bool(called[f])})
        print(f"  B repeat {rep} done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(frows)


def paired(df, key, metric):
    out = []
    for tag in TAGS:
        base = df[(df.embedding == tag) & (df.pooling == "mean")].set_index(key)[metric]
        for v in VARS[1:]:
            new = df[(df.embedding == tag) & (df.pooling == v)].set_index(key)[metric].reindex(base.index)
            diff = (new - base).dropna()
            p = wilcoxon(diff).pvalue if (diff != 0).sum() >= 3 else np.nan
            out.append({"embedding": tag, "pooling": v, "metric": metric, "mean_base": base.mean(),
                        "mean_new": new.mean(), "mean_diff": diff.mean(), "sd_diff": diff.std(),
                        "wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()),
                        "losses": int((diff < 0).sum()), "wilcoxon_p": p})
    return pd.DataFrame(out)


def main():
    d = load_rows("as_published")
    pools = {t: load_pools(t) for t in TAGS}
    d = d[d.sha1.isin(pools[TAGS[0]][1].keys())]
    a = part_a(d, pools)
    a.to_csv(P / f"results/pooling_repeats{SUF}_A.csv", index=False)
    b, bf = part_b(d, pools)
    b.to_csv(P / f"results/pooling_repeats{SUF}_B.csv", index=False)
    bf.to_csv(P / f"results/pooling_repeats{SUF}_B_fusions.csv", index=False)

    pa = pd.concat([paired(a, "seed", "AUROC"), paired(a, "seed", "AUPRC")])
    pb = pd.concat([paired(b, "repeat", "called"), paired(b, "repeat", "long_called")])
    long_f = bf[bf.long].pivot_table(index="fusion", columns=["embedding", "pooling"],
                                     values="called", aggfunc="mean").round(2)
    text = [f"# Pooling robustness (repeats){SUF}", "",
            "## A. Gene-disjoint, 10 gene assignments: paired vs whole-protein mean", "",
            pa.round(3).to_markdown(index=False), "",
            "## B. Paediatric panel, 5 calibration repeats: paired vs mean", "",
            pb.round(3).to_markdown(index=False), "",
            "Per-repeat counts:", "",
            b.groupby(["embedding", "pooling"])[["called", "long_called", "short_called", "cal_AUROC"]]
             .agg(["mean", "std", "min", "max"]).round(2).to_markdown(), "",
            "## B. Long fusions: share of the 5 repeats in which the fusion is called", "",
            long_f.to_markdown(), ""]
    (P / f"results/pooling_repeats{SUF}.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
