"""95% confidence intervals for the main results, from saved outputs only (no model is trained).

1. Benchmark (seq models + baselines; results/seq_model_predictions.csv.gz, results/baseline_predictions_*.csv.gz):
   AUROC / AUPRC with a cluster bootstrap over fusion pairs (hgene::tgene) of the test set, 2,000 draws.
   Paired differences on the same test rows: FusOn-pLM - ESM-2, and sequence MLP - gene-name baseline.
   Gene-disjoint "6 assignments" = mean over gene_disjoint (seed 42) + gd_s1..gd_s5, each bootstrapped
   independently (CI covers test sampling only; the across-assignment SD is reported beside it).
2. Pooling, gene-disjoint (results/pooling_repeats{,_up}_A.csv, 10 assignments): paired difference vs mean pooling
   and MANE+UniProt vs MANE, with a Nadeau-Bengio corrected t interval (assignments share training data, so the
   naive t interval is too narrow).
3. Paediatric panel (results/pooling_repeats{,_up}_B_fusions.csv, 5 calibration repeats): fusions called (/46) and
   long fusions called (/12), cluster bootstrap over fusions (10,000 draws), with paired differences.
Outputs: results/bootstrap_{benchmark,pooling_gd,panel}.csv, results/bootstrap_cis.md
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

P = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_seq_models import gene_disjoint_ids, load_rows  # noqa: E402

R = P / "results"
B_BENCH, B_PANEL = 2000, 10000
GD = ["gene_disjoint"] + [f"gd_s{k}" for k in range(1, 6)]
MAIN = ["random_row", "random_seq", "pair_grouped", "gene_disjoint"]


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return np.percentile(x, 2.5), np.percentile(x, 97.5)


def cluster_draws(groups, n_draws, seed):
    """List of row-index arrays, one per bootstrap draw, resampling clusters with replacement."""
    codes, uniq = pd.factorize(groups)
    members = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    for _ in range(n_draws):
        pick = rng.integers(0, len(members), len(members))
        yield np.concatenate([members[k] for k in pick])


def auc_pair(y, s):
    if y.min() == y.max():
        return np.nan, np.nan
    return roc_auc_score(y, s), average_precision_score(y, s)


def boot_scores(y, scores, groups, seed):
    """scores: dict name -> array aligned with y. Returns dict name -> (n_draws, 2) array of (AUROC, AUPRC)."""
    out = {k: [] for k in scores}
    for idx in cluster_draws(groups, B_BENCH, seed):
        for k, s in scores.items():
            out[k].append(auc_pair(y[idx], s[idx]))
    return {k: np.array(v) for k, v in out.items()}


def benchmark():
    seq = pd.read_csv(R / "seq_model_predictions.csv.gz")
    rows, diffs = [], []
    for labelset, sl in seq.groupby("labelset"):
        d = load_rows(labelset)
        pair = (d.hgene + "::" + d.tgene)
        base = pd.read_csv(R / f"baseline_predictions_{labelset}.csv.gz")
        for split in MAIN + GD[1:]:
            ss = sl[sl.split == split]
            wide = ss.pivot_table(index="row_id", columns=["embedding", "model"], values="score")
            wide.columns = [f"{e}/{m}" for e, m in wide.columns]
            if split in MAIN:
                bw = base[base.split == split].pivot_table(index="row_id", columns="model", values="score")
                bw.columns = [f"baseline/{c}" for c in bw.columns]
                wide = wide.join(bw, how="inner")
            wide = wide.dropna()
            y = d.loc[wide.index, "label"].to_numpy()
            g = pair.loc[wide.index].to_numpy()
            sc = {c: wide[c].to_numpy() for c in wide.columns if not c.endswith("/constant")}
            bs = boot_scores(y, sc, g, seed=zlib.crc32(f"{labelset}/{split}".encode()))
            for k, s in sc.items():
                au, ap = auc_pair(y, s)
                rows.append({"labelset": labelset, "split": split, "model": k, "n_test": len(y), "pos": int(y.sum()),
                             "pairs": len(set(g)), "AUROC": au, "AUROC_lo": ci(bs[k][:, 0])[0],
                             "AUROC_hi": ci(bs[k][:, 0])[1], "AUPRC": ap, "AUPRC_lo": ci(bs[k][:, 1])[0],
                             "AUPRC_hi": ci(bs[k][:, 1])[1], "_draws": bs[k]})
            comps = [("fuson_plm/mlp", "esm2_650m/mlp"), ("fuson_plm/logreg", "esm2_650m/logreg")]
            if split in MAIN:
                comps += [("esm2_650m/mlp", "baseline/gene_name"), ("fuson_plm/mlp", "baseline/gene_name"),
                          ("esm2_650m/mlp", "baseline/gene_role")]
            for a, b in comps:
                if a in sc and b in sc:
                    diffs.append({"labelset": labelset, "split": split, "comparison": f"{a} - {b}",
                                  "_a": bs[a], "_b": bs[b],
                                  "dAUROC": auc_pair(y, sc[a])[0] - auc_pair(y, sc[b])[0],
                                  "dAUPRC": auc_pair(y, sc[a])[1] - auc_pair(y, sc[b])[1]})
            print(f"  benchmark {labelset}/{split} done", flush=True)
    res = pd.DataFrame(rows)
    for r in diffs:
        dd = r["_a"] - r["_b"]
        r["dAUROC_lo"], r["dAUROC_hi"] = ci(dd[:, 0])
        r["dAUPRC_lo"], r["dAUPRC_hi"] = ci(dd[:, 1])
        r["P(dAUROC<=0)"] = float(np.nanmean(dd[:, 0] <= 0))
    dif = pd.DataFrame(diffs)
    # gene-disjoint: mean over the 6 assignments, draws averaged across independently bootstrapped assignments
    gdrows, gddif = [], []
    for (labelset, model), g in res[res.split.isin(GD)].groupby(["labelset", "model"]):
        if len(g) < 6:
            continue
        m = np.nanmean(np.stack(g._draws.to_list()), axis=0)
        gdrows.append({"labelset": labelset, "split": "gene_disjoint x6", "model": model, "n_test": int(g.n_test.mean()),
                       "pos": int(g.pos.mean()), "pairs": int(g.pairs.mean()), "AUROC": g.AUROC.mean(),
                       "AUROC_lo": ci(m[:, 0])[0], "AUROC_hi": ci(m[:, 0])[1], "AUROC_sd_assign": g.AUROC.std(),
                       "AUPRC": g.AUPRC.mean(), "AUPRC_lo": ci(m[:, 1])[0], "AUPRC_hi": ci(m[:, 1])[1]})
    for (labelset, comp), g in dif[dif.split.isin(GD)].groupby(["labelset", "comparison"]):
        if len(g) < 6:
            continue
        dd = np.nanmean(np.stack([a - b for a, b in zip(g._a, g._b)]), axis=0)
        gddif.append({"labelset": labelset, "split": "gene_disjoint x6", "comparison": comp,
                      "dAUROC": g.dAUROC.mean(), "dAUROC_lo": ci(dd[:, 0])[0], "dAUROC_hi": ci(dd[:, 0])[1],
                      "dAUPRC": g.dAUPRC.mean(), "dAUPRC_lo": ci(dd[:, 1])[0], "dAUPRC_hi": ci(dd[:, 1])[1],
                      "P(dAUROC<=0)": float(np.nanmean(dd[:, 0] <= 0)), "wins_of_6": int((g.dAUROC > 0).sum())})
    res = pd.concat([res.drop(columns="_draws"), pd.DataFrame(gdrows)], ignore_index=True)
    dif = pd.concat([dif.drop(columns=["_a", "_b"]), pd.DataFrame(gddif)], ignore_index=True)
    return res, dif


def nb_interval(diff, ratio):
    """Nadeau-Bengio corrected resampled t interval for the mean of per-split differences."""
    j = len(diff)
    se = np.sqrt((1 / j + ratio) * np.var(diff, ddof=1))
    t = stats.t.ppf(0.975, j - 1)
    p = 2 * stats.t.sf(abs(diff.mean()) / se, j - 1) if se > 0 else np.nan
    return diff.mean() - t * se, diff.mean() + t * se, p


def pooling_gd():
    d = load_rows("as_published")
    idx = pd.read_csv(P / "data/embeddings_pooled/esm2_650m/index.csv")
    d = d[d.sha1.isin(set(idx.seq_sha1))]
    sizes = [tuple(map(len, gene_disjoint_ids(d, seed=s))) for s in range(1, 11)]
    ratio = np.mean([te / tr for tr, va, te in sizes])
    a = pd.concat([pd.read_csv(R / "pooling_repeats_A.csv").assign(junctions="MANE"),
                   pd.read_csv(R / "pooling_repeats_up_A.csv").assign(junctions="MANE+UniProt")])
    a["embedding"] = a.embedding.str.replace("_up", "", regex=False)
    n_test_check = int(a[a.junctions == "MANE"].groupby("seed").n_test.first().sum()) == sum(s[2] for s in sizes)
    rows = []
    for metric in ("AUROC", "AUPRC"):
        w = a.pivot_table(index="seed", columns=["junctions", "embedding", "pooling"], values=metric)
        for j in ("MANE", "MANE+UniProt"):
            for e in ("esm2_650m", "fuson_plm"):
                for v in ("head+tail", "head+tail+junction"):
                    diff = (w[(j, e, v)] - w[(j, e, "mean")]).to_numpy()
                    lo, hi, p = nb_interval(diff, ratio)
                    rows.append({"metric": metric, "comparison": f"{v} - mean", "junctions": j, "embedding": e,
                                 "diff": diff.mean(), "lo": lo, "hi": hi, "p_corrected": p})
        for e in ("esm2_650m", "fuson_plm"):
            for v in ("head+tail", "head+tail+junction"):
                diff = (w[("MANE+UniProt", e, v)] - w[("MANE", e, v)]).to_numpy()
                lo, hi, p = nb_interval(diff, ratio)
                rows.append({"metric": metric, "comparison": "MANE+UniProt - MANE", "junctions": v, "embedding": e,
                             "diff": diff.mean(), "lo": lo, "hi": hi, "p_corrected": p})
        diff = (w[("MANE", "fuson_plm", "mean")] - w[("MANE", "esm2_650m", "mean")]).to_numpy()
        lo, hi, p = nb_interval(diff, ratio)
        rows.append({"metric": metric, "comparison": "FusOn-pLM - ESM-2 (mean pooling)", "junctions": "-",
                     "embedding": "-", "diff": diff.mean(), "lo": lo, "hi": hi, "p_corrected": p})
    return pd.DataFrame(rows), ratio, n_test_check


def panel():
    bf = pd.concat([pd.read_csv(R / "pooling_repeats_B_fusions.csv").assign(junctions="MANE"),
                    pd.read_csv(R / "pooling_repeats_up_B_fusions.csv").assign(junctions="MANE+UniProt")])
    bf["embedding"] = bf.embedding.str.replace("_up", "", regex=False)
    # per fusion: share of the 5 repeats in which it is called, for every configuration
    w = bf.pivot_table(index="fusion", columns=["junctions", "embedding", "pooling"], values="called", aggfunc="mean")
    long = bf.groupby("fusion")["long"].first().reindex(w.index).to_numpy()
    rng = np.random.default_rng(2026)
    out = []
    for subset, mask in (("all /46", np.ones(len(w), bool)), ("long /12", long)):
        m = w[mask].to_numpy()
        draws = m[rng.integers(0, len(m), (B_PANEL, len(m)))].sum(axis=1)  # (B, configs)
        col = {c: i for i, c in enumerate(w.columns)}

        def add(label, a, b=None):
            est = m[:, col[a]].sum() - (m[:, col[b]].sum() if b else 0)
            dd = draws[:, col[a]] - (draws[:, col[b]] if b else 0)
            lo, hi = ci(dd)
            out.append({"subset": subset, "quantity": label, "estimate": est, "lo": lo, "hi": hi,
                        **({"P(diff<=0)": float(np.mean(dd <= 0))} if b else {})})

        for j in ("MANE", "MANE+UniProt"):
            for e in ("esm2_650m", "fuson_plm"):
                for v in ("mean", "head+tail", "head+tail+junction"):
                    if j == "MANE+UniProt" and v == "mean":
                        continue  # identical to MANE
                    add(f"{e} {v} [{j}]", (j, e, v))
        add("fuson_plm - esm2_650m, mean pooling", ("MANE", "fuson_plm", "mean"), ("MANE", "esm2_650m", "mean"))
        for j in ("MANE", "MANE+UniProt"):
            for e in ("esm2_650m", "fuson_plm"):
                add(f"{e} head+tail - mean [{j}]", (j, e, "head+tail"), (j, e, "mean"))
        for e in ("esm2_650m", "fuson_plm"):
            add(f"{e} head+tail: UniProt - MANE", ("MANE+UniProt", e, "head+tail"), ("MANE", e, "head+tail"))
    return pd.DataFrame(out)


def fmt(x, lo, hi, nd=3):
    return f"{x:.{nd}f} ({lo:.{nd}f} to {hi:.{nd}f})"


def main():
    res, dif = benchmark()
    res.to_csv(R / "bootstrap_benchmark.csv", index=False)
    dif.to_csv(R / "bootstrap_benchmark_diffs.csv", index=False)
    pg, ratio, chk = pooling_gd()
    pg.to_csv(R / "bootstrap_pooling_gd.csv", index=False)
    pn = panel()
    pn.to_csv(R / "bootstrap_panel.csv", index=False)

    show = res[res.split.isin(MAIN[:3] + ["gene_disjoint x6"])].copy()
    show["AUROC (95% CI)"] = [fmt(*r) for r in show[["AUROC", "AUROC_lo", "AUROC_hi"]].to_numpy()]
    show["AUPRC (95% CI)"] = [fmt(*r) for r in show[["AUPRC", "AUPRC_lo", "AUPRC_hi"]].to_numpy()]
    dshow = dif[dif.split.isin(MAIN[:3] + ["gene_disjoint x6"])].copy()
    dshow["dAUROC (95% CI)"] = [fmt(*r) for r in dshow[["dAUROC", "dAUROC_lo", "dAUROC_hi"]].to_numpy()]
    dshow["dAUPRC (95% CI)"] = [fmt(*r) for r in dshow[["dAUPRC", "dAUPRC_lo", "dAUPRC_hi"]].to_numpy()]
    pg["diff (95% CI)"] = [fmt(*r) for r in pg[["diff", "lo", "hi"]].to_numpy()]
    pn["estimate (95% CI)"] = [fmt(*r, nd=1) for r in pn[["estimate", "lo", "hi"]].to_numpy()]
    text = ["# Bootstrap 95% confidence intervals (no retraining; saved predictions and per-split results only)", "",
            "## 1. Benchmark: cluster bootstrap over fusion pairs in each test set (2,000 draws)", "",
            "Gene-disjoint x6 = mean over 6 gene assignments; its CI covers test sampling only, and AUROC_sd_assign "
            "is the spread across assignments.", ""]
    for ls in ("as_published", "noncancer_fixed"):
        s = show[show.labelset == ls]
        text += [f"### {ls}", "", s[["split", "model", "n_test", "pos", "pairs", "AUROC (95% CI)", "AUPRC (95% CI)",
                                     "AUROC_sd_assign"]].round(3).to_markdown(index=False), "",
                 "Paired differences (same test rows):", "",
                 dshow[dshow.labelset == ls][["split", "comparison", "dAUROC (95% CI)", "dAUPRC (95% CI)",
                                              "P(dAUROC<=0)", "wins_of_6"]].round(3).to_markdown(index=False), ""]
    text += ["## 2. Pooling, gene-disjoint (10 assignments): Nadeau-Bengio corrected t intervals", "",
             f"Mean test/train ratio used for the correction: {ratio:.3f}. n_test matches pooling_repeats_A: {chk}.", "",
             pg[["metric", "comparison", "junctions", "embedding", "diff (95% CI)", "p_corrected"]].round(3)
             .to_markdown(index=False), "",
             "## 3. Paediatric panel: cluster bootstrap over fusions (10,000 draws), counts averaged over 5 repeats", "",
             pn[["subset", "quantity", "estimate (95% CI)", "P(diff<=0)"]].round(3).to_markdown(index=False), ""]
    (R / "bootstrap_cis.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
