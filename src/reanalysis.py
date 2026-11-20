"""Revision re-analysis of saved predictions (no training). Vectorised gene-level bootstrap.

Gene bootstrap: genes are drawn with replacement over the whole gene set (B draws); a test pair's weight is the
product of its two genes' counts; the same draw is applied to every gene assignment, and the AUROC is averaged
over assignments, so the interval reflects that assignments share genes and pairs. Weighted AUROC with ties = 1/2.

Parts (each writes results/reanalysis/<part>*.csv and a section of results/reanalysis/summary.md):
  fp       FusionPath-style benchmark (fp_rev predictions): every model on every split, paired differences
  v2       gene-role-free labels: main, family-held-out, 50-assignment and single-source runs; all / both genes
           unseen / no FusOn-DB sequence; models and paired differences
  pooling  Nadeau-Bengio corrected intervals with record-level and gene-level test/train ratios
  panel    fusion-resampled intervals for panel calls (both label sets) and paired differences
  overlap  FusOn-pLM training-split membership of benchmark sequences; FusOn-pLM minus ESM-2 by membership
Usage: python reanalysis.py [parts...]
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

P = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
R = P / "results"
OUT = R / "reanalysis"
B = 1000


# ------------------------------------------------------------------ vectorised weighted AUROC over bootstrap draws
def wauc_draws(y, s, W):
    """y [n] 0/1, s [n] scores, W [B, n] weights -> AUROC per draw [B] (nan if a class has zero weight)."""
    o = np.argsort(s, kind="mergesort")
    s, y, W = s[o], y[o], W[:, o]
    _, start = np.unique(s, return_index=True)
    Wp, Wn = W * (y == 1), W * (y == 0)
    Pg, Ng = np.add.reduceat(Wp, start, axis=1), np.add.reduceat(Wn, start, axis=1)
    below = np.cumsum(Ng, axis=1) - Ng
    num = (Pg * (below + 0.5 * Ng)).sum(1)
    den = Wp.sum(1) * Wn.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def gene_boot(blocks, models, genes, b=B, seed=0):
    """blocks: list of DataFrames (one per assignment) with columns label, hgene, tgene and one score column per
    model. Returns point [M], draws [b, M] (mean over assignments), per-block point [K, M]."""
    gi = {g: i for i, g in enumerate(genes)}
    rng = np.random.default_rng(seed)
    C = rng.multinomial(len(genes), np.full(len(genes), 1 / len(genes)), size=b).astype(np.float32)
    draws = np.zeros((b, len(models)))
    cnt = np.zeros((b, len(models)))
    per = []
    for blk in blocks:
        y = blk.label.to_numpy()
        h = np.array([gi[g] for g in blk.hgene]); t = np.array([gi[g] for g in blk.tgene])
        W = C[:, h] * C[:, t]
        row = []
        for j, m in enumerate(models):
            s = blk[m].to_numpy(dtype=float)
            row.append(wauc_draws(y, s, np.ones((1, len(y))))[0])
            a = wauc_draws(y, s, W)
            ok = ~np.isnan(a)
            draws[ok, j] += a[ok]; cnt[ok, j] += 1
        per.append(row)
    per = np.array(per)
    return np.nanmean(per, 0), draws / np.maximum(cnt, 1), per


def summarise(point, draws, per, models, comps, extra):
    rows = []
    lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    j = {m: i for i, m in enumerate(models)}
    for m in models:
        rows.append({**extra, "kind": "model", "name": m, "AUROC": point[j[m]], "ci_lo": lo[j[m]], "ci_hi": hi[j[m]],
                     "sd_over_assignments": np.nanstd(per[:, j[m]], ddof=1) if len(per) > 1 else np.nan, "n_assignments": len(per)})
    for a, b_ in comps:
        if a in j and b_ in j:
            d = draws[:, j[a]] - draws[:, j[b_]]
            l2, h2 = np.nanpercentile(d, [2.5, 97.5])
            rows.append({**extra, "kind": "difference", "name": f"{a} - {b_}", "AUROC": point[j[a]] - point[j[b_]],
                         "ci_lo": l2, "ci_hi": h2, "p_boot_le0": float(np.nanmean(d <= 0)),
                         "wins": f"{int((per[:, j[a]] > per[:, j[b_]]).sum())}/{len(per)}", "n_assignments": len(per)})
    return rows


# ------------------------------------------------------------------ FusionPath-style benchmark
def part_fp():
    pr = pd.concat([pd.read_csv(f) for f in sorted((R / "fp_rev").glob("predictions_*.csv.gz"))])
    rows = []
    for labelset, q in pr.groupby("labelset"):
        genes = sorted(set(q.hgene) | set(q.tgene))
        models = sorted(q.model.unique())
        wide = q.pivot_table(index=["split", "row_id", "hgene", "tgene", "label"], columns="model", values="score").reset_index()
        comps = [(m, "gene_name") for m in models if m not in ("gene_name", "gene_role")] + \
                [("fuson_plm/mlp", "esm2_650m/mlp"), ("fuson_plm_native/mlp", "fuson_plm/mlp"), ("esm2_3b/mlp", "esm2_650m/mlp"),
                 ("prot_t5/mlp", "esm2_650m/mlp"), ("fusionpath_like/mlp", "fuson_plm_native/mlp"),
                 ("fuson_plm/logreg", "esm2_650m/logreg"), ("kmer3", "esm2_650m/mlp"), ("homology", "esm2_650m/mlp")]
        groups = {"random_row": ["random_row"], "random_seq": ["random_seq"], "pair_grouped": ["pair_grouped"],
                  "genes_held_out": [f"genes_held_out_{k}" for k in range(1, 11)]}
        for gname, splits in groups.items():
            blocks = [wide[wide.split == sp].dropna(subset=models) for sp in splits if (wide.split == sp).any()]
            if not blocks:
                continue
            pt, dr, per = gene_boot(blocks, models, genes)
            rows += summarise(pt, dr, per, models, comps, {"dataset": "fp", "labelset": labelset, "split": gname})
            print(f"fp {labelset}/{gname}: {len(blocks)} blocks", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ gene-role-free labels
def part_v2():
    import train_v2 as T
    import train_v2_checks as Ck
    rows = []
    s = None
    runs = [("main", "gene", ["predictions.csv.gz", "predictions_checks.csv.gz"]),
            ("family", "family", ["predictions.csv.gz", "predictions_checks.csv.gz"]),
            ("seeds50", "gene", ["predictions.csv.gz", "predictions_checks.csv.gz"]),
            ("matched", "gene", ["predictions.csv.gz"])]
    for name, mode, files in runs:
        d = R / "v2_rev" / name
        fs = [d / f for f in files if (d / f).exists()]
        if not fs:
            print(f"v2 {name}: no predictions yet", flush=True)
            continue
        T.SPLIT_MODE = mode
        if s is None:
            s, _ = T.load()
            prov = Ck.provenance(s, s)
            sj = s.join(prov, on=["pair", "sha1"])
            fz = sj.groupby("pair").fuson_db.max()
            fusondb_free = set(fz.index[fz == 0])
            genes_of = dict(zip(s.pair, zip(s.hgene, s.tgene)))
            genes = sorted(set(s.hgene) | set(s.tgene))
        pr = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop(columns=["both_unseen"], errors="ignore")
        pr = T.mark_both_unseen(pr, s)
        pr["hgene"] = pr.pair.map(lambda p: genes_of[p][0]); pr["tgene"] = pr.pair.map(lambda p: genes_of[p][1])
        tags = [k for k in T.TAGS]
        for sub, q in pr.groupby("subset"):
            models = sorted(q.model.unique())
            comps = [(f"{k}_logreg", "shortcuts") for k in tags] + [(f"{k}_mlp", "shortcuts") for k in tags] + \
                    [(f"{k}+shortcuts", "shortcuts") for k in tags] + [(f"{k}_logreg", "shortcuts+source") for k in tags] + \
                    [(f"{k}+shortcuts+source", "shortcuts+source") for k in tags] + \
                    [("kmer3", "shortcuts"), ("shortcuts+source", "shortcuts"), ("source", "shortcuts"), ("fuson_logreg", "esm2_logreg"),
                     ("fuson_native_logreg", "fuson_logreg"), ("esm2_3b_logreg", "esm2_logreg"), ("prot_t5_logreg", "esm2_logreg"),
                     ("fuson_mlp", "esm2_mlp")]
            for label, filt in [("all test pairs", None), ("both genes unseen", "both"), ("no FusOn-DB sequence", "fz")]:
                qq = q if filt is None else q[q.both_unseen.astype(bool)] if filt == "both" else q[q.pair.isin(fusondb_free)]
                wide = qq.pivot_table(index=["seed", "pair", "hgene", "tgene", "label"], columns="model", values="p").reset_index()
                blocks = [g.dropna(subset=models) for _, g in wide.groupby("seed")]
                blocks = [b_ for b_ in blocks if 0 < b_.label.sum() < len(b_)]
                if not blocks:
                    continue
                pt, dr, per = gene_boot(blocks, models, genes)
                rows += summarise(pt, dr, per, models, comps,
                                  {"dataset": f"v2_{name}", "subset": sub, "test_pairs": label,
                                   "median_test_pairs": float(np.median([len(b_) for b_ in blocks])),
                                   "median_test_pos": float(np.median([b_.label.sum() for b_ in blocks]))})
            print(f"v2 {name}/{sub} done", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ pooling: Nadeau-Bengio with gene-level ratio
def part_pooling():
    import train_seq_models as Mq
    d = Mq.load_rows("as_published")
    sizes = []
    for k in range(1, 11):
        tr, va, te = Mq.gene_disjoint_ids(d, seed=k)
        gtr = set(d.loc[tr, "hgene"]) | set(d.loc[tr, "tgene"]); gte = set(d.loc[te, "hgene"]) | set(d.loc[te, "tgene"])
        sizes.append({"seed": k, "n_train": len(tr), "n_test": len(te), "genes_train": len(gtr), "genes_test": len(gte)})
    sz = pd.DataFrame(sizes)
    r_rec = (sz.n_test / sz.n_train).mean()
    r_gene = (sz.genes_test / sz.genes_train).mean()
    rows = []
    for suf, junc in [("", "MANE"), ("_up", "MANE + UniProt")]:
        f = R / f"pooling_repeats{suf}_A.csv"
        if not f.exists():
            continue
        a = pd.read_csv(f)
        for emb, q in a.groupby("embedding"):
            w = q.pivot_table(index="seed", columns="pooling", values="AUROC")
            for v in [c for c in w.columns if c != "mean"]:
                dlt = (w[v] - w["mean"]).dropna().to_numpy()
                J, mu, var = len(dlt), dlt.mean(), dlt.var(ddof=1)
                tq = stats.t.ppf(0.975, J - 1)
                for rname, ratio in [("record-level", r_rec), ("gene-level", r_gene)]:
                    half = tq * np.sqrt((1 / J + ratio) * var)
                    rows.append({"dataset": "pooling", "junctions": junc, "embedding": emb, "comparison": f"{v} - mean",
                                 "ratio": rname, "test_train_ratio": ratio, "mean_diff": mu, "ci_lo": mu - half, "ci_hi": mu + half,
                                 "wins": f"{int((dlt > 0).sum())}/{J}", "mean_AUROC": w["mean"].mean(), "variant_AUROC": w[v].mean()})
    sz.to_csv(OUT / "pooling_split_sizes.csv", index=False)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ panels: fusion-resampled intervals
def panel_table(fus, label, models_pairs, b=10000):
    """fus: model, repeat, fusion, called (bool), median_len. Called share per fusion averaged over repeats."""
    sh = fus.groupby(["model", "fusion"]).called.mean().unstack(0)
    length = fus.groupby("fusion").median_len.first().reindex(sh.index)
    long = (length > 1500).to_numpy()
    rng = np.random.default_rng(1)
    idx = rng.integers(0, len(sh), size=(b, len(sh)))
    rows = []
    for m in sh.columns:
        v = sh[m].to_numpy()
        tot = v[idx].sum(1); lg = (v * long)[idx].sum(1)
        rows.append({"dataset": f"panel_{label}", "kind": "model", "name": m, "called": v.sum(),
                     "ci_lo": np.percentile(tot, 2.5), "ci_hi": np.percentile(tot, 97.5), "long_called": (v * long).sum(),
                     "long_ci_lo": np.percentile(lg, 2.5), "long_ci_hi": np.percentile(lg, 97.5), "n_fusions": len(v), "n_long": int(long.sum())})
    for a, b_ in models_pairs:
        if a in sh and b_ in sh:
            dv = (sh[a] - sh[b_]).to_numpy()
            tot = dv[idx].sum(1); lg = (dv * long)[idx].sum(1)
            rows.append({"dataset": f"panel_{label}", "kind": "difference", "name": f"{a} - {b_}", "called": dv.sum(),
                         "ci_lo": np.percentile(tot, 2.5), "ci_hi": np.percentile(tot, 97.5), "long_called": (dv * long).sum(),
                         "long_ci_lo": np.percentile(lg, 2.5), "long_ci_hi": np.percentile(lg, 97.5), "n_fusions": len(dv), "n_long": int(long.sum())})
    return rows


def part_panel():
    rows = []
    f = R / "fp_rev/panel/fusion_calls.csv"
    if f.exists():
        fus = pd.read_csv(f)
        pairs = [("fuson_plm/mlp", "esm2_650m/mlp"), ("fuson_plm/logreg", "esm2_650m/logreg"), ("fuson_plm_native/mlp", "fuson_plm/mlp"),
                 ("esm2_3b/mlp", "esm2_650m/mlp"), ("prot_t5/mlp", "esm2_650m/mlp"), ("fusionpath_like/mlp", "fuson_plm_native/mlp"),
                 ("esm2_650m/mlp", "gene_name"), ("fusionpath_like/mlp", "gene_name")]
        rows += panel_table(fus, "fusionpath_labels", pairs)
    f = R / "v2_strict/panel/panel_scores.csv"
    if f.exists():
        iso = pd.read_csv(f)
        ps = pd.read_csv(P / "derived/panel_isoforms.csv")
        seqlen = {}
        with open(P / "derived/panel_isoforms.fasta") as fh:
            lines = fh.read().split("\n")
        for i in range(0, len(lines) - 1, 2):
            if lines[i].startswith(">"):
                seqlen[lines[i][1:]] = len(lines[i + 1])
        iso["length"] = iso.sha1.map(seqlen)
        fus = iso.groupby(["model", "repeat", "fusion"]).agg(share=("called", "mean"), median_len=("length", "median")).reset_index()
        fus["called"] = fus.share > 0.5
        pairs = [("fuson_mlp", "esm2_mlp"), ("fuson_logreg", "esm2_logreg"), ("esm2_mlp", "shortcuts"), ("fuson_mlp", "shortcuts"),
                 ("esm2+shortcuts", "shortcuts"), ("shortcuts", "gene_name"), ("fuson_native_mlp", "fuson_mlp"),
                 ("esm2_3b_mlp", "esm2_mlp"), ("prot_t5_mlp", "esm2_mlp")]
        rows += panel_table(fus, "gene_role_free_labels", pairs)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ FusOn-pLM pretraining overlap with the benchmarks
def part_overlap():
    import train_v2 as T
    from embed_sequences import labelled_sequences
    snap = next((P / "models/hf_cache/hub/models--ChatterjeeLab--FusOn-pLM/snapshots").iterdir())
    split_of = {}
    for sp in ("train", "val", "test"):
        for q in pd.read_csv(snap / f"fuson_plm/data/splits/{sp}_df.csv", usecols=["sequence"]).sequence:
            split_of[hashlib.sha1(str(q).replace("*", "").upper().encode()).hexdigest()] = sp
    rows = []
    T.SPLIT_MODE = "gene"
    s, _ = T.load()
    s["fuson_split"] = s.sha1.map(split_of).fillna("none")
    tab = s.drop_duplicates("sha1").groupby(["label", "fuson_split"]).size().unstack(fill_value=0)
    tab.to_csv(OUT / "overlap_v2_sequences.csv")
    pair_in_train = s.groupby("pair").fuson_split.apply(lambda x: (x == "train").any())
    f = [R / "v2_rev/main/predictions.csv.gz"]
    if f[0].exists():
        pr = pd.read_csv(f[0])
        genes_of = dict(zip(s.pair, zip(s.hgene, s.tgene)))
        pr["hgene"] = pr.pair.map(lambda p: genes_of[p][0]); pr["tgene"] = pr.pair.map(lambda p: genes_of[p][1])
        genes = sorted(set(s.hgene) | set(s.tgene))
        for sub in ["AB_all", "A_only"]:
            q = pr[pr.subset == sub]
            for grp, keep in [("pair has a FusOn-pLM training sequence", True), ("pair has none", False)]:
                qq = q[q.pair.map(pair_in_train).fillna(False) == keep]
                models = ["esm2_logreg", "fuson_logreg", "fuson_native_logreg", "esm2_mlp", "fuson_mlp"]
                models = [m for m in models if m in set(qq.model)]
                wide = qq.pivot_table(index=["seed", "pair", "hgene", "tgene", "label"], columns="model", values="p").reset_index()
                blocks = [g.dropna(subset=models) for _, g in wide.groupby("seed")]
                blocks = [b_ for b_ in blocks if 0 < b_.label.sum() < len(b_)]
                if not blocks:
                    continue
                pt, dr, per = gene_boot(blocks, models, genes)
                rows += summarise(pt, dr, per, models, [("fuson_logreg", "esm2_logreg"), ("fuson_mlp", "esm2_mlp"),
                                                        ("fuson_native_logreg", "esm2_logreg")],
                                  {"dataset": "overlap_v2", "subset": sub, "group": grp,
                                   "median_test_pairs": float(np.median([len(b_) for b_ in blocks])),
                                   "median_test_pos": float(np.median([b_.label.sum() for b_ in blocks]))})
    # FusionPath-style benchmark: share of test records whose sequence is in FusOn-pLM's training split
    fp = pd.concat([pd.read_csv(f, usecols=["labelset", "split", "model", "row_id", "label", "score"])
                    for f in sorted((R / "fp_rev").glob("predictions_*.csv.gz"))])
    import fp_benchmark_rev as Bm
    d = Bm.load("as_published")
    mem = d.sha1.map(split_of).fillna("none")
    rows.append({"dataset": "overlap_fp", "name": "labelled records by FusOn-pLM split",
                 **{f"share_{k}": v for k, v in mem.value_counts(normalize=True).round(4).items()}})
    g = fp[(fp.labelset == "as_published") & fp.split.str.startswith("genes_held_out")].copy()
    g["in_train"] = g.row_id.map(mem) == "train"
    for (model, it), q in g.groupby(["model", "in_train"]):
        if model in ("esm2_650m/mlp", "fuson_plm/mlp", "fuson_plm_native/mlp", "esm2_650m/logreg", "fuson_plm/logreg"):
            au = q.groupby("split").apply(lambda z: stats.mannwhitneyu(z.score[z.label == 1], z.score[z.label == 0]).statistic
                                          / max(1, (z.label == 1).sum() * (z.label == 0).sum()) if z.label.nunique() == 2 else np.nan)
            rows.append({"dataset": "overlap_fp", "name": model, "group": "in FusOn-pLM training split" if it else "not in training split",
                         "AUROC": au.mean(), "sd_over_assignments": au.std(), "n_assignments": int(au.notna().sum())})
    return pd.DataFrame(rows)


def main():
    parts = sys.argv[1:] or ["fp", "v2", "pooling", "panel", "overlap"]
    OUT.mkdir(parents=True, exist_ok=True)
    fn = {"fp": part_fp, "v2": part_v2, "pooling": part_pooling, "panel": part_panel, "overlap": part_overlap}
    for p in parts:
        t0 = time.time()
        df = fn[p]()
        df.to_csv(OUT / f"{p}.csv", index=False)
        (OUT / f"{p}.md").write_text(f"# Re-analysis: {p}\n\n" + df.round(4).to_markdown(index=False) + "\n")
        print(f"== {p}: {len(df)} rows in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
