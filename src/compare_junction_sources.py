"""Paired comparison of pooling results with MANE-only junctions vs MANE + UniProt junctions.

Same gene assignments (seeds 1..10), MLP seeds and calibration repeats (101..105) in both runs, so differences
are paired. Inputs: results/pooling_repeats_{A,B,B_fusions}.csv and results/pooling_repeats_up_{...}.csv.
Output: results/junction_source_comparison.md
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

P = F2P_ROOT / "results"


def load(name):
    old = pd.read_csv(P / f"pooling_repeats_{name}.csv").assign(junctions="MANE")
    new = pd.read_csv(P / f"pooling_repeats_up_{name}.csv").assign(junctions="MANE+UniProt")
    new["embedding"] = new.embedding.str.replace("_up", "", regex=False)
    return pd.concat([old, new])


def paired(df, key, metric):
    out = []
    for (emb, pool), g in df.groupby(["embedding", "pooling"]):
        w = g.pivot_table(index=key, columns="junctions", values=metric)
        d = (w["MANE+UniProt"] - w["MANE"]).dropna()
        p = wilcoxon(d).pvalue if (d != 0).sum() >= 3 else np.nan
        out.append({"embedding": emb, "pooling": pool, "metric": metric, "MANE": w["MANE"].mean(),
                    "MANE+UniProt": w["MANE+UniProt"].mean(), "diff": d.mean(),
                    "up": int((d > 0).sum()), "down": int((d < 0).sum()), "wilcoxon_p": p})
    return pd.DataFrame(out)


def main():
    a, b, bf = load("A"), load("B"), load("B_fusions")
    # sanity: mean pooling ignores junctions, so it must be identical between runs
    m = a[a.pooling == "mean"].pivot_table(index=["embedding", "seed"], columns="junctions", values="AUROC")
    mean_check = float((m["MANE"] - m["MANE+UniProt"]).abs().max())
    pa = pd.concat([paired(a, "seed", "AUROC"), paired(a, "seed", "AUPRC")])
    pb = pd.concat([paired(b, "repeat", "called"), paired(b, "repeat", "long_called")])
    # head+tail vs mean within each junction source (gene-disjoint AUROC)
    rows = []
    for (j, emb), g in a.groupby(["junctions", "embedding"]):
        base = g[g.pooling == "mean"].set_index("seed").AUROC
        for v in ("head+tail", "head+tail+junction"):
            d = (g[g.pooling == v].set_index("seed").AUROC - base).dropna()
            rows.append({"junctions": j, "embedding": emb, "pooling": v, "AUROC_vs_mean": d.mean(),
                         "wins_of_10": int((d > 0).sum()), "wilcoxon_p": wilcoxon(d).pvalue})
    vs_mean = pd.DataFrame(rows)
    focus = ["KIAA1549::BRAF", "KMT2A::AFDN", "KMT2A::AFF1", "KMT2A::ELL", "KMT2A::EPS15", "KMT2A::MLLT1",
             "KMT2A::MLLT10", "KMT2A::MLLT3", "RUNX1::RUNX1T1", "PAX3::FOXO1", "PAX7::FOXO1", "TPM3::NTRK1",
             "NUP98::NSD1", "DEK::NUP214", "SET::NUP214"]
    ff = bf[bf.fusion.isin(focus)].pivot_table(index="fusion", columns=["embedding", "pooling", "junctions"],
                                               values="called", aggfunc="mean").round(1)
    text = ["# Junction source: MANE only vs MANE + UniProt (paired, same seeds)", "",
            f"Sanity check: mean pooling ignores junctions; max |AUROC difference| between runs = {mean_check:.4f}.", "",
            "## A. Gene-disjoint, 10 gene assignments", "", pa.round(3).to_markdown(index=False), "",
            "Partner-aware pooling vs whole-protein mean, within each junction source:", "",
            vs_mean.round(3).to_markdown(index=False), "",
            "## B. Paediatric panel, 5 calibration repeats (fusions called /46)", "", pb.round(2).to_markdown(index=False), "",
            "## B. Selected fusions: share of 5 repeats called", "", ff.to_markdown(), ""]
    (P / "junction_source_comparison.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
