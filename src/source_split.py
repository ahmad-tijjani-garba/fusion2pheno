"""Round-3 review: split the data-source features. Logistic regression only; CPU.

The data-source features of train_v2_checks.py mix three kinds of information:
  where the sequence came from  A_pipeline (our translation vs database sequence), fusionpath_db, fuson_db, fusiongdb2
  breakpoint origin             bp_fusionpath, bp_singh2020  (Singh et al. 2020 Table S3 is also a NEGATIVE label source)
  study intensity               log_nseq (number of distinct sequences of the pair)
Variants (each alone, with the shortcut features, and with shortcuts + ESM-2 / FusOn-pLM):
  src_full    all of the above (as in the manuscript)
  src_db      sequence origin only (A_pipeline and database membership)
  src_nosingh all except bp_singh2020
  src_nocount all except log_nseq
Pooled sources: AB_all and AB_nonRT, seeds 1-50 (as results/v2_rev/seeds50).
Single sources: the four source-matched sets, seeds 42 and 1-20 (as results/v2_rev/matched).
Gene-resampled 95% CIs (reanalysis.gene_boot). Outputs in results/round3/.
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

P = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reanalysis as RA  # noqa: E402
import train_v2 as T  # noqa: E402
import train_v2_checks as C  # noqa: E402
import train_v2_matched as M  # noqa: E402

OUT = P / ("results/round3_quick" if "--quick" in sys.argv else "results/round3")
OUT.mkdir(parents=True, exist_ok=True)
SRC_ALL = C.SRC + ["log_nseq"]
VAR = {"src_full": SRC_ALL,
       "src_db": ["A_pipeline", "fusionpath_db", "fuson_db", "fusiongdb2"],
       "src_nosingh": [c for c in SRC_ALL if c != "bp_singh2020"],
       "src_nocount": list(C.SRC)}


def feats(name, d, vocab, E):
    """name: '<variant>', 'shortcuts+<variant>' or '<emb>+shortcuts+<variant>'."""
    v = name.split("+")[-1]
    b = []
    if "shortcuts" in name:
        b.append(T.features("shortcuts", d, vocab, None))
    b.append(d[VAR[v]].to_numpy(dtype=np.float32))
    if E is not None:
        b.append(E)
    return np.hstack(b)


JOBS = ([(v, None) for v in VAR] + [(f"shortcuts+{v}", None) for v in VAR]
        + [(f"esm2+shortcuts+{v}", "esm2") for v in ("src_db", "src_nosingh", "src_nocount")]
        + [(f"fuson+shortcuts+{v}", "fuson") for v in ("src_db", "src_nosingh")])


def fit_all(d0, allpairs, sub, seeds, emb, min_class=2):
    preds = []
    for seed in seeds:
        t0 = time.time()
        part, _ = T.split(allpairs, seed)
        d = d0.assign(part=d0.pair.map(part))
        test_seqs = set(d.sha1[d.part == "test"])
        tr = d[(d.part == "train") & ~d.sha1.isin(test_seqs)]
        va = d[(d.part == "val") & ~d.sha1.isin(test_seqs)]
        te = d[d.part == "test"]
        n = [(x.drop_duplicates("pair").label == c).sum() for x in (tr, va, te) for c in (0, 1)]
        if min(n) < min_class:
            continue
        vocab = {g: j for j, g in enumerate(sorted(set(tr.hgene) | set(tr.tgene)))}
        ytr, wtr = tr.label.to_numpy(), T.weights(tr)
        for name, k in JOBS:
            F = [feats(name, x, vocab, emb[k][x.index.to_numpy()] if k else None) for x in (tr, va, te)]
            p, Cc = T.fit_logreg(F[0], ytr, wtr, F[1], va, F[2])
            preds.append(T.pair_scores(te, p).assign(subset=sub, seed=seed, model=name, C=Cc))
        print(f"{sub} seed {seed}: {time.time() - t0:.0f}s", flush=True)
    return preds


def main():
    quick = "--quick" in sys.argv  # 2 seeds per set, to test the pipeline
    t0 = time.time()
    s, emb = T.load()
    s = s.join(C.provenance(s, s), on=["pair", "sha1"])
    allpairs = s.drop_duplicates("pair")[["pair", "hgene", "tgene"]]
    genes_of = dict(zip(s.pair, zip(s.hgene, s.tgene)))
    genes = sorted(set(s.hgene) | set(s.tgene))

    # which features carry the label directly: share of pairs with each flag, by class
    pairs = s.groupby("pair").agg(label=("label", "first"), **{c: (c, "max") for c in C.SRC},
                                  nseq=("sha1", "nunique"))
    tab = pairs.groupby("label")[C.SRC].mean().T.rename(columns={0: "share_of_negative_pairs", 1: "share_of_positive_pairs"})
    tab["n_pairs_with_flag"] = pairs[C.SRC].sum().astype(int)
    tab.loc["median_nseq", ["share_of_negative_pairs", "share_of_positive_pairs"]] = pairs.groupby("label").nseq.median().to_numpy()
    tab.to_csv(OUT / "source_features_by_label.csv")
    print(tab.round(3).to_string(), flush=True)

    preds = []
    for sub in ("AB_all", "AB_nonRT"):
        d0 = (s if sub == "AB_all" else s[s.rt == 0]).copy()
        d0["log_nseq"] = np.log10(d0.groupby("pair").sha1.transform("nunique"))
        preds += fit_all(d0, allpairs, sub, (list(range(1, 51)) if not quick else [1, 2]), emb)
    for sub in M.SUBSETS:
        d0 = M.subset(s, sub).copy()
        d0["log_nseq"] = np.log10(d0.groupby("pair").sha1.transform("nunique"))
        preds += fit_all(d0, allpairs, sub, ([42, *range(1, 21)] if not quick else [42, 1]), emb)
    new = pd.concat(preds).reset_index()
    new.to_csv(OUT / "source_split_predictions.csv.gz", index=False)

    # merge with the manuscript's predictions for the same assignments (shortcuts, esm2/fuson + shortcuts, src_full baseline)
    old = pd.concat([pd.read_csv(P / "results/v2_rev/seeds50/predictions.csv.gz"),
                     pd.read_csv(P / "results/v2_rev/seeds50/predictions_checks.csv.gz"),
                     pd.read_csv(P / "results/v2_rev/matched/predictions.csv.gz")], ignore_index=True)
    old = old[old.model.isin(["shortcuts", "esm2+shortcuts", "fuson+shortcuts", "source", "shortcuts+source",
                              "esm2+shortcuts+source", "fuson+shortcuts+source"])]
    old = old.drop(columns=[c for c in old.columns if c not in ("pair", "label", "p", "subset", "seed", "model")])
    pr = pd.concat([old, new[["pair", "label", "p", "subset", "seed", "model"]]], ignore_index=True)
    pr = T.mark_both_unseen(pr.drop(columns=["both_unseen"], errors="ignore"), s)
    pr["hgene"] = pr.pair.map(lambda x: genes_of[x][0]); pr["tgene"] = pr.pair.map(lambda x: genes_of[x][1])

    models = (["shortcuts", "source", "shortcuts+source", "esm2+shortcuts", "fuson+shortcuts", "esm2+shortcuts+source",
               "fuson+shortcuts+source"] + [n for n, _ in JOBS])
    comps = ([(v, "shortcuts") for v in ["source"] + list(VAR)]
             + [(v, "source") for v in VAR]
             + [("esm2+shortcuts", "shortcuts"), ("fuson+shortcuts", "shortcuts"),
                ("esm2+shortcuts+source", "shortcuts+source"), ("fuson+shortcuts+source", "shortcuts+source")]
             + [(f"esm2+shortcuts+{v}", f"shortcuts+{v}") for v in ("src_db", "src_nosingh", "src_nocount")]
             + [(f"fuson+shortcuts+{v}", f"shortcuts+{v}") for v in ("src_db", "src_nosingh")]
             + [(f"shortcuts+{v}", "shortcuts") for v in VAR])
    rows = []
    for sub, q in pr.groupby("subset"):
        for label, filt in [("all test pairs", None), ("both genes unseen", "both")]:
            z = q if filt is None else q[q.both_unseen.astype(bool)]
            ms = [m for m in models if m in set(z.model)]
            wide = z.pivot_table(index=["seed", "pair", "hgene", "tgene", "label"], columns="model", values="p").reset_index()
            blocks = [b.dropna(subset=ms) for _, b in wide.groupby("seed")]
            blocks = [b for b in blocks if 0 < b.label.sum() < len(b)]
            if not blocks:
                continue
            pt, dr, per = RA.gene_boot(blocks, ms, genes)
            rows += RA.summarise(pt, dr, per, ms, comps, {
                "subset": sub, "test_pairs": label, "n_assignments": len(blocks),
                "median_test_pairs": float(np.median([len(b) for b in blocks])),
                "median_test_neg": float(np.median([(b.label == 0).sum() for b in blocks]))})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "source_split.csv", index=False)
    keep = res[res.name.isin(["source", "src_db", "src_nosingh", "src_nocount", "src_full", "shortcuts"])
               | res.name.str.contains(" - ")]
    md = ["# Round 3: data-source features split\n", tab.round(3).to_markdown(), ""]
    for (sub, tp), g in keep.groupby(["subset", "test_pairs"]):
        md += [f"## {sub}: {tp} ({int(g.n_assignments.iloc[0])} assignments, median {g.median_test_pairs.iloc[0]:.0f} "
               f"test pairs, {g.median_test_neg.iloc[0]:.0f} negative)", "",
               g[["kind", "name", "AUROC", "ci_lo", "ci_hi", "wins"]].round(3).to_markdown(index=False), ""]
    (OUT / "source_split.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
