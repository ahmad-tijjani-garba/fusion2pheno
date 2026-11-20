"""External-review major revisions. CPU only.

Part A (comments 1-2, trains logistic regressions): a provenance- and layout-matched gene-role-free set.
  Read-through pairs removed; within strata of inter-chromosomal status x quartile of pair length x sequence origin
  (all our translations / all database sequences / mixed), equal numbers of positive and negative pairs are kept
  (random, seed 0). Then length, layout, gene names, shortcut features, data-source features, ESM-2 and FusOn-pLM
  (alone, with the shortcut features, reduced to 32 principal components with the shortcut features, and with the
  shortcut and data-source features), on the 50 pooled gene assignments. If matching works, length, layout and
  data source should score near 0.5.
Part B (comment 7, no training): AUPRC (gene-resampled 95% CIs), prevalence and sensitivity at 90% specificity
  (read from each test ROC) from the saved predictions: FusionPath-style splits and gene-role-free pooled and single
  sources.
Part C (comment 4, no training): two-level bootstrap (gene assignments resampled with replacement, then genes) for
  the main comparisons, compared with the gene-only bootstrap used in the manuscript.
Outputs in results/round3/: matched*.csv, metrics_auprc.csv, bootstrap_two_level.csv, round3b.md
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_curve

P = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reanalysis as RA  # noqa: E402
import train_v2 as T  # noqa: E402
import train_v2_checks as C  # noqa: E402

QUICK = "--quick" in sys.argv
OUT = P / ("results/round3_quick" if QUICK else "results/round3")
OUT.mkdir(parents=True, exist_ok=True)
B = 200 if QUICK else 1000
md = ["# External review: matched set, AUPRC and two-level bootstrap\n"]


# ------------------------------------------------------------------ helpers
def blocks_from(wide, keys, models):
    bl = [b.dropna(subset=models) for _, b in wide.groupby(keys)]
    return [b for b in bl if 0 < b.label.sum() < len(b)]


def gene_counts(genes, b, seed=0):
    rng = np.random.default_rng(seed)
    return rng.multinomial(len(genes), np.full(len(genes), 1 / len(genes)), size=b).astype(np.float32)


def auprc_boot(blocks, models, genes, b=B):
    gi = {g: i for i, g in enumerate(genes)}
    Cn = gene_counts(genes, b)
    draws = np.zeros((b, len(models))); cnt = np.zeros((b, len(models)))
    point, prev, sens = [], [], []
    for blk in blocks:
        y = blk.label.to_numpy()
        h = np.array([gi[g] for g in blk.hgene]); t = np.array([gi[g] for g in blk.tgene])
        W = Cn[:, h] * Cn[:, t]
        prev.append(y.mean())
        row, srow = [], []
        for j, m in enumerate(models):
            s = blk[m].to_numpy(dtype=float)
            row.append(average_precision_score(y, s))
            fpr, tpr, _ = roc_curve(y, s)
            srow.append(float(np.interp(0.10, fpr, tpr)))
            for k in range(b):
                w = W[k]
                if w[y == 1].sum() > 0 and w[y == 0].sum() > 0:
                    draws[k, j] += average_precision_score(y, s, sample_weight=w); cnt[k, j] += 1
        point.append(row); sens.append(srow)
    draws = draws / np.maximum(cnt, 1)
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return (np.mean(point, 0), lo, hi, float(np.mean(prev)), np.mean(sens, 0))


def two_level(blocks, models, genes, comps, b=B, seed=1):
    """Gene-only vs two-level (assignments, then genes) percentile intervals of mean AUROC and paired differences."""
    gi = {g: i for i, g in enumerate(genes)}
    Cn = gene_counts(genes, b)
    A = np.full((len(blocks), b, len(models)), np.nan)
    for i, blk in enumerate(blocks):
        y = blk.label.to_numpy()
        h = np.array([gi[g] for g in blk.hgene]); t = np.array([gi[g] for g in blk.tgene])
        W = Cn[:, h] * Cn[:, t]
        for j, m in enumerate(models):
            A[i, :, j] = RA.wauc_draws(y, blk[m].to_numpy(dtype=float), W)
    gene_only = np.nanmean(A, axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(blocks), size=(b, len(blocks)))
    two = np.stack([np.nanmean(A[idx[k], k, :], axis=0) for k in range(b)])
    j = {m: i for i, m in enumerate(models)}
    rows = []
    for a, c in comps:
        if a in j and c in j:
            for name, D in (("gene only", gene_only), ("two-level", two)):
                d = D[:, j[a]] - D[:, j[c]]
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                rows.append({"comparison": f"{a} - {c}", "bootstrap": name, "ci_lo": lo, "ci_hi": hi, "width": hi - lo})
    return rows


# ------------------------------------------------------------------ Part A: matched set
def part_a(s, emb):
    d = s[s.rt == 0].copy()
    d["log_nseq"] = np.log10(d.groupby("pair").sha1.transform("nunique"))
    g = d.groupby("pair").agg(label=("label", "first"), inter=("inter", "first"), loglen=("log_len", "median"),
                              trans=("A_pipeline", "mean"))
    g["lenbin"] = pd.qcut(g.loglen, 4, labels=False, duplicates="drop")
    g["origin"] = np.where(g.trans == 1, "translated", np.where(g.trans == 0, "database", "mixed"))
    rng = np.random.default_rng(0)
    keep, strata = [], []
    for key, q in g.groupby(["inter", "lenbin", "origin"]):
        pos, neg = q.index[q.label == 1].to_numpy(), q.index[q.label == 0].to_numpy()
        n = min(len(pos), len(neg))
        strata.append({"inter": key[0], "length_quartile": key[1], "origin": key[2], "positive_pairs": len(pos),
                       "negative_pairs": len(neg), "kept_per_class": n})
        if n:
            keep += list(rng.choice(pos, n, replace=False)) + list(rng.choice(neg, n, replace=False))
    st = pd.DataFrame(strata)
    st.to_csv(OUT / "matched_strata.csv", index=False)
    dm = d[d.pair.isin(set(keep))]
    print(f"matched set: {len(keep)} pairs ({len(keep) // 2} per class), {len(dm)} sequences", flush=True)
    allpairs = s.drop_duplicates("pair")[["pair", "hgene", "tgene"]]
    jobs = ["length", "geometry", "gene_name", "shortcuts", "source", "esm2_logreg", "fuson_logreg", "esm2+shortcuts",
            "fuson+shortcuts", "esm2_pca32+shortcuts", "fuson_pca32+shortcuts", "esm2+shortcuts+source",
            "fuson+shortcuts+source"]
    preds = []
    for seed in ([1, 2] if QUICK else range(1, 51)):
        part, _ = T.split(allpairs, seed)
        x = dm.assign(part=dm.pair.map(part))
        test_seqs = set(x.sha1[x.part == "test"])
        tr = x[(x.part == "train") & ~x.sha1.isin(test_seqs)]
        va = x[(x.part == "val") & ~x.sha1.isin(test_seqs)]
        te = x[x.part == "test"]
        n = [(z.drop_duplicates("pair").label == c).sum() for z in (tr, va, te) for c in (0, 1)]
        if min(n) < 2:
            continue
        vocab = {gg: k for k, gg in enumerate(sorted(set(tr.hgene) | set(tr.tgene)))}
        ytr, wtr = tr.label.to_numpy(), T.weights(tr)
        for name in jobs:
            k = name.split("_")[0].split("+")[0]
            E = [emb[k][z.index.to_numpy()] for z in (tr, va, te)] if k in emb else None
            if name in ("length", "geometry", "gene_name", "shortcuts"):
                F = [T.features(name, z, vocab, None) for z in (tr, va, te)]
            elif name == "source":
                F = [C.feats("source", z, vocab, None) for z in (tr, va, te)]
            elif name.endswith("_logreg"):
                F = E
            elif name.endswith("+shortcuts+source"):
                F = [C.feats("shortcuts+source", z, vocab, E[i]) for i, z in enumerate((tr, va, te))]
            elif "_pca32" in name:
                pca = PCA(n_components=32, random_state=0).fit(E[0])
                F = [np.hstack([T.features("shortcuts", z, vocab, None), pca.transform(E[i])]) for i, z in enumerate((tr, va, te))]
            else:
                F = [T.features("plm+shortcuts", z, vocab, E[i]) for i, z in enumerate((tr, va, te))]
            p, Cc = T.fit_logreg(F[0], ytr, wtr, F[1], va, F[2])
            preds.append(T.pair_scores(te, p).assign(subset="matched", seed=seed, model=name, C=Cc))
        print(f"matched seed {seed}: train {n[0]}/{n[1]} val {n[2]}/{n[3]} test {n[4]}/{n[5]}", flush=True)
    pr = T.mark_both_unseen(pd.concat(preds).reset_index(), s)
    genes_of = dict(zip(s.pair, zip(s.hgene, s.tgene)))
    pr["hgene"] = pr.pair.map(lambda z: genes_of[z][0]); pr["tgene"] = pr.pair.map(lambda z: genes_of[z][1])
    pr.to_csv(OUT / "matched_predictions.csv.gz", index=False)
    genes = sorted(set(s.hgene) | set(s.tgene))
    comps = [(m, "shortcuts") for m in jobs if m not in ("shortcuts",)] + [
        ("esm2+shortcuts+source", "shortcuts"), ("esm2_logreg", "fuson_logreg")]
    rows = []
    for label, filt in (("all test pairs", None), ("both genes unseen", True)):
        q = pr if filt is None else pr[pr.both_unseen.astype(bool)]
        wide = q.pivot_table(index=["seed", "pair", "hgene", "tgene", "label"], columns="model", values="p").reset_index()
        bl = blocks_from(wide, "seed", jobs)
        if not bl:
            continue
        pt, dr, per = RA.gene_boot(bl, jobs, genes)
        rows += RA.summarise(pt, dr, per, jobs, comps, {"test_pairs": label, "n_assignments": len(bl),
                                                        "median_test_pairs": float(np.median([len(x) for x in bl])),
                                                        "median_test_neg": float(np.median([(x.label == 0).sum() for x in bl]))})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "matched.csv", index=False)
    md.extend(["## A. Matched set", "", f"{len(keep)} pairs kept ({len(keep) // 2} per class) of "
               f"{int((g.label == 1).sum())} positive and {int((g.label == 0).sum())} negative non-read-through pairs.", "",
               st.to_markdown(index=False), "", res.round(3).to_markdown(index=False), ""])


# ------------------------------------------------------------------ Part B and C
def fp_blocks():
    pr = pd.concat([pd.read_csv(f) for f in sorted((P / "results/fp_rev").glob("predictions_*.csv.gz"))]
                   + [pd.read_csv(P / "results/fp_rev_fpr/predictions_FPR.csv.gz")])
    pr = pr[pr.labelset == "as_published"]
    return pr


def parts_bc(s):
    rows_b, rows_c = [], []
    # FusionPath-style
    pr = fp_blocks()
    fg = sorted(set(pr.hgene) | set(pr.tgene))
    fp_models = ["gene_name", "gene_role", "kmer3", "homology", "esm2_650m/mlp", "esm2_3b/mlp", "prot_t5/mlp",
                 "fuson_plm/mlp", "fuson_plm_native/mlp", "fusionpath_like/mlp", "fusionpath_rebuild/full", "domain_retention"]
    groups = {"random_row": ["random_row"], "random_seq": ["random_seq"], "pair_grouped": ["pair_grouped"],
              "genes_held_out": [f"genes_held_out_{k}" for k in range(1, 11)]}
    for gname, sps in groups.items():
        q = pr[pr.split.isin(sps)]
        ms = [m for m in fp_models if m in set(q.model)]
        wide = q.pivot_table(index=["split", "row_id", "hgene", "tgene", "label"], columns="model", values="score").reset_index()
        bl = blocks_from(wide, "split", ms)
        pt, lo, hi, prev, sens = auprc_boot(bl, ms, fg)
        for i, m in enumerate(ms):
            rows_b.append({"benchmark": "FusionPath-style", "set": gname, "model": m, "AUPRC": pt[i], "ci_lo": lo[i],
                           "ci_hi": hi[i], "prevalence": prev, "sens_at_90spec": sens[i], "n_assignments": len(bl)})
        if gname == "genes_held_out":
            rows_c += [{"benchmark": "FusionPath-style held-out genes", **r} for r in two_level(bl, ms, fg, [
                ("esm2_650m/mlp", "gene_name"), ("esm2_650m/mlp", "kmer3"), ("fuson_plm/mlp", "esm2_650m/mlp"),
                ("esm2_3b/mlp", "esm2_650m/mlp"), ("fusionpath_like/mlp", "fuson_plm_native/mlp"),
                ("fusionpath_rebuild/full", "fusionpath_like/mlp"), ("domain_retention", "esm2_650m/mlp")])]
    # gene-role-free
    genes = sorted(set(s.hgene) | set(s.tgene))
    genes_of = dict(zip(s.pair, zip(s.hgene, s.tgene)))
    v = pd.concat([pd.read_csv(P / "results/v2_rev/seeds50/predictions.csv.gz"),
                   pd.read_csv(P / "results/v2_rev/seeds50/predictions_checks.csv.gz"),
                   pd.read_csv(P / "results/v2_rev/matched/predictions.csv.gz")], ignore_index=True)
    v["hgene"] = v.pair.map(lambda z: genes_of[z][0]); v["tgene"] = v.pair.map(lambda z: genes_of[z][1])
    v2_models = ["gene_name", "length", "geometry", "kmer3", "shortcuts", "source", "esm2_logreg", "fuson_logreg",
                 "prot_t5_logreg", "esm2_3b_logreg", "esm2+shortcuts", "fuson+shortcuts", "esm2_mlp", "fuson_mlp"]
    for sub, q in v.groupby("subset"):
        ms = [m for m in v2_models if m in set(q.model)]
        wide = q.pivot_table(index=["seed", "pair", "hgene", "tgene", "label"], columns="model", values="p").reset_index()
        bl = blocks_from(wide, "seed", ms)
        if not bl:
            continue
        pt, lo, hi, prev, sens = auprc_boot(bl, ms, genes)
        for i, m in enumerate(ms):
            rows_b.append({"benchmark": "gene-role-free", "set": sub, "model": m, "AUPRC": pt[i], "ci_lo": lo[i],
                           "ci_hi": hi[i], "prevalence": prev, "sens_at_90spec": sens[i], "n_assignments": len(bl)})
        rows_c += [{"benchmark": f"gene-role-free {sub}", **r} for r in two_level(bl, ms, genes, [
            ("esm2+shortcuts", "shortcuts"), ("fuson+shortcuts", "shortcuts"), ("esm2_logreg", "shortcuts"),
            ("source", "shortcuts"), ("fuson_logreg", "esm2_logreg")])]
        print(f"metrics {sub} done", flush=True)
    b = pd.DataFrame(rows_b); c = pd.DataFrame(rows_c)
    b.to_csv(OUT / "metrics_auprc.csv", index=False)
    c.to_csv(OUT / "bootstrap_two_level.csv", index=False)
    w = c.pivot_table(index=["benchmark", "comparison"], columns="bootstrap", values="width").reset_index()
    w["ratio_two_level_to_gene"] = w["two-level"] / w["gene only"]
    md.extend(["## B. AUPRC, prevalence and sensitivity at 90% specificity", "", b.round(3).to_markdown(index=False), "",
               "## C. Two-level vs gene-only bootstrap (95% interval widths)", "", w.round(3).to_markdown(index=False), "",
               c.round(3).to_markdown(index=False), ""])


def main():
    t0 = time.time()
    s, emb = T.load()
    s = s.join(C.provenance(s, s), on=["pair", "sha1"])
    part_a(s, emb)
    parts_bc(s)
    (OUT / "round3b.md").write_text("\n".join(md) + "\n")
    print("\n".join(md)[:6000])
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
