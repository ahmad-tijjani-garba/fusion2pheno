"""Follow-up checks on the strict label set v2 results.

1. Sequence-source baseline. Where a sequence came from (our GENCODE pipeline from FusionPath or Singh 2020
   breakpoints; FusionPath / FusOn-DB / FusionGDB2 database sequences) and how many distinct sequences a pair has
   are label-free properties of the data set that could separate the classes. New logreg models:
     source, shortcuts+source, esm2+shortcuts+source, fuson+shortcuts+source
   Same splits, weights, C selection and pair-level scoring as train_v2.py.
2. Gene-level bootstrap. Genes are resampled with replacement over the whole gene universe; a pair's weight is
   the product of its two genes' multiplicities. The same gene draw is applied to all 11 gene assignments, so the
   CI of the mean-over-assignments AUROC reflects that the assignments share genes and pairs.
3. FusOn-DB-free test pairs: AUROC on test pairs none of whose sequences occur in FusOn-DB (FusOn-pLM's training
   source), using the same predictions.
Outputs: results/v2_strict/{predictions_checks.csv.gz, gene_bootstrap.csv, checks.md}
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import train_v2 as T

OUT = T.P / "results/v2_strict"
SRC = ["A_pipeline", "bp_fusionpath", "bp_singh2020", "fusionpath_db", "fuson_db", "fusiongdb2"]


def provenance(s, d0):
    raw = pd.read_csv(T.P / "data/labels_v2/strict/sequences.csv.gz", usecols=["pair", "sha1", "set", "bp_sources"], low_memory=False)
    raw = raw[raw.set_index(["pair", "sha1"]).index.isin(s.set_index(["pair", "sha1"]).index)]
    srcs = raw.bp_sources.fillna("").str.split(";")
    f = pd.DataFrame({"pair": raw.pair, "sha1": raw.sha1,
                      "A_pipeline": (raw.set == "A_pipeline").astype(float),
                      "bp_fusionpath": ((raw.set == "A_pipeline") & srcs.map(lambda x: "fusionpath" in x)).astype(float),
                      "bp_singh2020": srcs.map(lambda x: "singh2020" in x).astype(float),
                      "fusionpath_db": srcs.map(lambda x: "fusionpath_db" in x).astype(float),
                      "fuson_db": srcs.map(lambda x: "fuson_db" in x).astype(float),
                      "fusiongdb2": srcs.map(lambda x: "fusiongdb2" in x).astype(float)})
    f = f.groupby(["pair", "sha1"]).max()
    return f


def add_source(d0, prov):
    d = d0.join(prov, on=["pair", "sha1"])
    d["log_nseq"] = np.log10(d.groupby("pair").sha1.transform("nunique"))
    return d


def feats(name, d, vocab, E):
    b = []
    if "shortcuts" in name:
        b += [T.features("shortcuts", d, vocab, None)]
    if "source" in name:
        b += [d[SRC + ["log_nseq"]].to_numpy(dtype=np.float32)]
    if E is not None:
        b += [E]
    return np.hstack(b)


def run(s, emb, prov, subsets, seeds):
    allpairs = s.drop_duplicates("pair")[["pair", "hgene", "tgene"]]
    preds = []
    for sub in subsets:
        d0 = s if sub == "AB_all" else s[s.rt == 0] if sub == "AB_nonRT" else s[s.set == "A_pipeline"]
        d0 = add_source(d0, prov)
        for seed in seeds:
            t0 = time.time()
            part, _ = T.split(allpairs, seed)
            d = d0.assign(part=d0.pair.map(part))
            test_seqs = set(d.sha1[d.part == "test"])
            tr = d[(d.part == "train") & ~d.sha1.isin(test_seqs)]
            va = d[(d.part == "val") & ~d.sha1.isin(test_seqs)]
            te = d[d.part == "test"]
            if min(tr.label.nunique(), va.label.nunique(), te.label.nunique()) < 2:
                continue
            vocab = {g: j for j, g in enumerate(sorted(set(tr.hgene) | set(tr.tgene)))}
            ytr, wtr = tr.label.to_numpy(), T.weights(tr)
            for name, k in [("source", None), ("shortcuts+source", None),
                            ("esm2+shortcuts+source", "esm2"), ("fuson+shortcuts+source", "fuson")]:
                F = [feats(name, x, vocab, emb[k][x.index.to_numpy()] if k else None) for x in (tr, va, te)]
                p, C = T.fit_logreg(F[0], ytr, wtr, F[1], va, F[2])
                preds.append(T.pair_scores(te, p).assign(subset=sub, seed=seed, model=name, C=C))
            print(f"{sub} seed {seed}: {time.time() - t0:.0f}s", flush=True)
    return pd.concat(preds).reset_index()


# ------------------------------------------------------------------ gene bootstrap
def wauc(y, p, w):
    pos = y == 1
    p0, w0 = p[~pos], w[~pos]
    o = np.argsort(p0)
    p0, w0 = p0[o], w0[o]
    c0 = np.concatenate([[0.0], np.cumsum(w0)])
    p1, w1 = p[pos], w[pos]
    lo, hi = np.searchsorted(p0, p1, "left"), np.searchsorted(p0, p1, "right")
    den = w1.sum() * w0.sum()
    return np.nan if den == 0 else (w1 * (c0[lo] + 0.5 * (c0[hi] - c0[lo]))).sum() / den


def gene_bootstrap(pr, genes_of, sub, models, keep=None, B=1000, seed=0, both=False):
    q = pr[pr.subset == sub]
    if keep is not None:
        q = q[q.pair.isin(keep)]
    if both:  # revision: only test pairs whose two genes are both unseen in that assignment
        q = q[q.both_unseen.astype(bool)]
    genes = sorted({g for hg in genes_of.values() for g in hg})
    gi = {g: i for i, g in enumerate(genes)}
    blocks = []
    for sd in sorted(q.seed.unique()):
        w = q[q.seed == sd].pivot_table(index="pair", columns="model", values="p")[models]
        lab = q[q.seed == sd].groupby("pair").label.first().reindex(w.index).to_numpy()
        hi = np.array([gi[genes_of[p][0]] for p in w.index])
        ti = np.array([gi[genes_of[p][1]] for p in w.index])
        if 0 < lab.sum() < len(lab):  # an assignment whose kept test pairs hold one class only is left out
            blocks.append((lab, w.to_numpy(), hi, ti))
    if not blocks:
        return None
    one = np.ones(len(genes))
    point = np.mean([[wauc(y, P[:, j], one[h] * one[t]) for j in range(len(models))] for y, P, h, t in blocks], axis=0)
    per_seed = np.array([[wauc(y, P[:, j], one[h] * one[t]) for j in range(len(models))] for y, P, h, t in blocks])
    rng = np.random.default_rng(seed)
    boot = []
    while len(boot) < B:
        m = np.bincount(rng.integers(0, len(genes), len(genes)), minlength=len(genes)).astype(float)
        vals = []
        for y, P, h, t in blocks:
            wt = m[h] * m[t]
            if wt[y == 1].sum() == 0 or wt[y == 0].sum() == 0:
                vals = None
                break
            vals.append([wauc(y, P[:, j], wt) for j in range(len(models))])
        if vals is not None:
            boot.append(np.mean(vals, axis=0))
    return point, np.array(boot), per_seed, sum(len(b[0]) for b in blocks) / len(blocks), \
        np.mean([b[0].sum() for b in blocks])


def report(pr, s, fusondb_free, subsets):
    genes_of = dict(zip(s.pair, zip(s.hgene, s.tgene)))
    base = ["length", "geometry", "role", "gene_name", "kmer3", "source", "shortcuts", "shortcuts+source"]
    present = set(pr.model)
    models_all = [m for m in base if m in present] + sorted(m for m in present if m not in base)
    comps = ([(f"{k}_logreg", "shortcuts") for k in T.TAGS] + [(f"{k}_mlp", "shortcuts") for k in T.TAGS]
             + [(f"{k}+shortcuts", "shortcuts") for k in T.TAGS] + [(f"{k}_logreg", "shortcuts+source") for k in T.TAGS]
             + [(f"{k}+shortcuts+source", "shortcuts+source") for k in T.TAGS]
             + [("kmer3", "shortcuts"), ("shortcuts+source", "shortcuts"), ("fuson_logreg", "esm2_logreg"),
                ("fuson_native_logreg", "fuson_logreg"), ("fuson_native_logreg", "esm2_logreg"),
                ("esm2_3b_logreg", "esm2_logreg"), ("prot_t5_logreg", "esm2_logreg")])
    L = ["# Strict label set v2: follow-up checks\n",
         "AUROC = mean over the gene-held-out assignments (pair-level). **Gene CI**: 95% CI from resampling genes "
         "(1,000 draws, same draw for every assignment). 'wins' = assignments where a > b.\n",
         "Source features: our pipeline vs database sequence; breakpoint from FusionPath or Singh 2020; present in "
         "FusionPath / FusOn-DB / FusionGDB2; log number of distinct sequences of the pair.\n"]
    rows = []
    for label, keep, both in [("all test pairs", None, False), ("both genes unseen", None, True),
                              ("test pairs with no FusOn-DB sequence", fusondb_free, False)]:
        for sub in subsets:
            models = [m for m in models_all if m in set(pr[pr.subset == sub].model)]
            res = gene_bootstrap(pr, genes_of, sub, models, keep=keep, both=both)
            if res is None:
                continue
            point, boot, ps, npairs, npos = res
            lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
            L.append(f"## {sub}: {label} (median {npairs:.0f} test pairs, {npos:.0f} positive, per assignment)\n")
            L.append(pd.DataFrame({"AUROC": point, "gene CI": [f"{a:.3f}-{b:.3f}" for a, b in zip(lo, hi)],
                                   "sd over assignments": ps.std(axis=0, ddof=1)}, index=models).round(3).to_markdown() + "\n")
            j = {m: i for i, m in enumerate(models)}
            D = []
            for a, b in comps:
                if a in j and b in j:
                    diff = boot[:, j[a]] - boot[:, j[b]]
                    lo_, hi_ = np.percentile(diff, [2.5, 97.5])
                    D.append({"comparison": f"{a} - {b}", "dAUROC": point[j[a]] - point[j[b]],
                              "gene CI": f"{lo_:.3f} to {hi_:.3f}", "P(boot <= 0)": (diff <= 0).mean(),
                              "wins": f"{int((ps[:, j[a]] > ps[:, j[b]]).sum())}/{len(ps)}"})
                    rows.append({"test_pairs": label, "subset": sub, **D[-1], "ci_lo": lo_, "ci_hi": hi_})
            L.append(pd.DataFrame(D).round(3).to_markdown(index=False) + "\n")
            for m in models:
                rows.append({"test_pairs": label, "subset": sub, "comparison": m, "dAUROC": point[j[m]],
                             "ci_lo": lo[j[m]], "ci_hi": hi[j[m]]})
    return "\n".join(L), pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=T.SEEDS)
    ap.add_argument("--subsets", nargs="*", default=["AB_all", "AB_nonRT", "A_only"])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    s, emb = T.load()
    prov = provenance(s, s)
    print("source x label (unique sequences):")
    print(s.join(prov, on=["pair", "sha1"]).groupby("label")[SRC].sum().astype(int).to_string(), flush=True)
    new = run(s, emb, prov, a.subsets, a.seeds)
    new.to_csv(out / "predictions_checks.csv.gz", index=False)
    old = pd.read_csv(out / "predictions.csv.gz")
    old = old[old.subset.isin(a.subsets) & old.seed.isin(a.seeds)]
    pr = T.mark_both_unseen(pd.concat([old.drop(columns=["both_unseen"], errors="ignore"), new], ignore_index=True), s)
    fz = s.join(prov, on=["pair", "sha1"]).groupby("pair").fuson_db.max()
    md, tab = report(pr, s, set(fz.index[fz == 0]), a.subsets)
    tab.to_csv(out / "gene_bootstrap.csv", index=False)
    (out / "checks.md").write_text(md)
    print(md)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
