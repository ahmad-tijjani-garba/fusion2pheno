"""Compare pooling strategies (mean, max, head+tail, head+tail+junction) for sequence-only models.

Part A: benchmark splits (as-published labels): random_seq, pair_grouped, gene_disjoint
        (seed 42 + 5 more gene assignments). MLP 1280k->256->1, 3 seeds averaged.
Part B: paediatric panel, as in score_panel.py: panel records removed from training,
        85/15 pair-grouped train/calibration, call = >= 90%-specificity threshold.
Outputs: results/pooling_benchmark.csv, results/pooling_panel.csv, results/pooling_summary.md
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

P = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusondb_paediatric_coverage import PANEL, names  # noqa: E402
from train_seq_models import gene_disjoint_ids, load_rows, metrics, norm_seq, train_mlp  # noqa: E402

TAGS = ["esm2_650m", "fuson_plm"]
VARIANTS = {"mean": ["mean"], "max": ["max"], "head+tail": ["head", "tail"],
            "head+tail+junction": ["head", "tail", "junction"]}
SEEDS = (0, 1, 2)


def load_pools(tag):
    z = np.load(P / f"data/embeddings_pooled/{tag}/pools.npz")
    idx = pd.read_csv(P / f"data/embeddings_pooled/{tag}/index.csv")
    lut = dict(zip(idx.seq_sha1, range(len(idx))))
    return {k: z[k].astype(np.float32) for k in z.files}, lut, idx


def feats(pools, lut, sha1s, variant):
    rows = [lut[h] for h in sha1s]
    return np.hstack([pools[p][rows] for p in VARIANTS[variant]])


def mlp_scores(Xtr, ytr, Xva, yva, Xte):
    sc = StandardScaler().fit(Xtr)
    a, b, c = [sc.transform(x).astype(np.float32) for x in (Xtr, Xva, Xte)]
    return np.mean([train_mlp(a, ytr, b, yva, c, s) for s in SEEDS], axis=0)


def part_a(d, pools_by_tag):
    split_sets = {}
    for s in ["random_seq", "pair_grouped", "gene_disjoint"]:
        sd = P / "data/splits/as_published" / s
        split_sets[s] = [np.intersect1d(pd.read_csv(sd / f"{x}.csv").row_id, d.index) for x in ("train", "val", "test")]
    for k in range(1, 6):
        split_sets[f"gd_s{k}"] = gene_disjoint_ids(d, seed=k)
    rows = []
    for split, (tr, va, te) in split_sets.items():
        y = [d.loc[i, "label"].to_numpy() for i in (tr, va, te)]
        for tag, (pools, lut, _) in pools_by_tag.items():
            for v in VARIANTS:
                X = [feats(pools, lut, d.loc[i, "sha1"], v) for i in (tr, va, te)]
                p = mlp_scores(X[0], y[0], X[1], y[1], X[2])
                au, ap = metrics(y[2], p)
                rows.append({"split": split, "embedding": tag, "pooling": v, "n_test": len(te),
                             "pos": int(y[2].sum()), "AUROC": au, "AUPRC": ap})
        print(f"  A {split} done", flush=True)
    return pd.DataFrame(rows)


def part_b(d, pools_by_tag):
    panel_pairs = {(hh, tt) for h, t, _ in PANEL for hh in names(h) for tt in names(t)}
    d = d[[(h, t) not in panel_pairs for h, t in zip(d.hgene, d.tgene)]]
    rng = np.random.default_rng(42)
    pairs = (d.hgene + "::" + d.tgene).unique()
    cal_pairs = set(rng.choice(pairs, size=int(0.15 * len(pairs)), replace=False))
    in_cal = (d.hgene + "::" + d.tgene).isin(cal_pairs).to_numpy()
    tr, cal = d[~in_cal], d[in_cal]
    ytr, ycal = tr.label.to_numpy(), cal.label.to_numpy()

    prev = pd.read_csv(P / "results/panel_scores.csv")[["fusion", "disease", "sha1"]]  # same 482 isoforms
    rows = []
    for tag, (pools, lut, idx) in pools_by_tag.items():
        lengths = dict(zip(idx.seq_sha1, idx.length))
        jsrc = dict(zip(idx.seq_sha1, idx.junction_source))
        for v in VARIANTS:
            Xtr = feats(pools, lut, tr.sha1, v)
            Xcal = feats(pools, lut, cal.sha1, v)
            Xp = feats(pools, lut, prev.sha1, v)
            both = mlp_scores(Xtr, ytr, Xcal, ycal, np.vstack([Xcal, Xp]))
            cs, ps = both[:len(Xcal)], both[len(Xcal):]
            thr = np.quantile(cs[ycal == 0], 0.90)
            g = prev.assign(score=ps, call=ps >= thr, length=prev.sha1.map(lengths),
                            junction_source=prev.sha1.map(jsrc))
            for fusion, gg in g.groupby("fusion", sort=False):
                rows.append({"embedding": tag, "pooling": v, "cal_AUROC": roc_auc_score(ycal, cs),
                             "fusion": fusion, "disease": gg.disease.iloc[0], "isoforms": len(gg),
                             "median_len": gg.length.median(), "called_share": gg.call.mean(),
                             "junction_found_share": (gg.junction_source != "none").mean()})
        print(f"  B {tag} done", flush=True)
    return pd.DataFrame(rows)


def main():
    d = load_rows("as_published")
    pools_by_tag = {t: load_pools(t) for t in TAGS}
    d = d[d.sha1.isin(pools_by_tag[TAGS[0]][1].keys())]
    a = part_a(d, pools_by_tag)
    a.to_csv(P / "results/pooling_benchmark.csv", index=False)
    b = part_b(d, pools_by_tag)
    b.to_csv(P / "results/pooling_panel.csv", index=False)

    a["family"] = np.where(a.split.str.match(r"^(gene_disjoint|gd_s\d)$"), "gene_disjoint (6)", a.split)
    tab_a = a.groupby(["family", "embedding", "pooling"]).agg(
        AUROC=("AUROC", "mean"), AUROC_sd=("AUROC", "std"), AUPRC=("AUPRC", "mean")).round(3).reset_index()
    b["called"] = b.called_share >= 0.5
    b["long"] = b.median_len > 1500
    tab_b = b.groupby(["embedding", "pooling"]).agg(
        cal_AUROC=("cal_AUROC", "first"), called=("called", "sum"), n=("called", "size"),
        long_called=("called", lambda s: int(s[b.loc[s.index, "long"]].sum())),
        long_n=("long", "sum")).reset_index()
    long_tab = b[b.long].pivot_table(index="fusion", columns=["embedding", "pooling"],
                                     values="called_share").round(2)
    text = ["# Pooling comparison", "",
            "## A. Benchmark splits (MLP, 3 seeds; gene_disjoint = mean over 6 gene assignments)", "",
            tab_a.to_markdown(index=False), "",
            "## B. Paediatric panel: fusions called pathogenic (majority of isoforms, 90% specificity)", "",
            tab_b.to_markdown(index=False), "",
            "## B. Long fusions (>1,500 aa): share of isoforms called", "",
            long_tab.to_markdown(), ""]
    (P / "results/pooling_summary.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
