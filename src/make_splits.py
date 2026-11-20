"""Build train/val/test splits of the rebuilt FusionPath dataset and measure leakage.

Splits (labelled rows only: Benign + Pathogenic):
(numpy-only; group sizes are by group count, so row proportions are approximate)
  random_row   - random rows, 70/15/15 (FusionPath 02_split_train_test_val.py)
  random_seq   - unique protein sequences at random, 60/20/20 (FusionPath split_generate_embeddings.py)
  pair_grouped - all records of a head::tail pair kept together, 70/15/15
  gene_disjoint- each gene assigned to one split; a fusion is kept only if both partners
                 are in the same split, so test genes never occur in training

Output per split: data/splits/<labelset>/<split>/{train,val,test}.csv with
row_id, hgene, tgene, label, plus results/splits_<labelset>.md with leakage stats.

Usage: python make_splits.py [as_published|noncancer_fixed]
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42


def train_test_split(items, test_size, random_state):
    """numpy stand-in for sklearn's train_test_split (same proportions, own RNG)."""
    items = np.asarray(items)
    perm = np.random.default_rng(random_state).permutation(len(items))
    n_test = int(np.ceil(test_size * len(items)))
    return items[perm[n_test:]], items[perm[:n_test]]


def group_split(index, groups, test_size, random_state):
    """Split index so that all members of a group land on the same side."""
    uniq = pd.unique(groups)
    _, test_groups = train_test_split(uniq, test_size, random_state)
    in_test = np.isin(groups, test_groups)
    return index[~in_test], index[in_test]
P = F2P_ROOT
M = P / "data/fusionpath_rebuild/01_source_data/merged"
FILES = {
    "as_published": "merged_fusions.filtered.hg38.formatted.labeled.csv",
    "noncancer_fixed": "merged_fusions.filtered.hg38.formatted.noncancerfix.labeled.csv",
}


def load(labelset):
    d = pd.read_csv(M / FILES[labelset], low_memory=False,
                    usecols=["hgene", "tgene", "aa_seq", "cancer_label", "primary_source", "cancer"])
    d["row_id"] = np.arange(len(d))
    d = d[d["cancer_label"].isin(["Benign", "Pathogenic"])].copy()
    d["label"] = (d["cancer_label"] == "Pathogenic").astype(int)
    n_missing = int((d["hgene"].isna() | d["tgene"].isna()).sum())
    if n_missing:
        print(f"note: {n_missing} labelled rows lack a head or tail gene symbol; kept with gene '?'")
    d["hgene"] = d["hgene"].fillna("?").astype(str).str.upper()
    d["tgene"] = d["tgene"].fillna("?").astype(str).str.upper()
    d["pair"] = d["hgene"] + "::" + d["tgene"]
    return d


def random_row(d):
    tr, tmp = train_test_split(d.index, test_size=0.30, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, random_state=SEED)
    return tr, va, te


def random_seq(d):
    seqs = d["aa_seq"].astype(str).str.slice(stop=2000).unique()  # FusionPath truncates first
    trv, te = train_test_split(seqs, test_size=0.20, random_state=SEED)
    tr, va = train_test_split(trv, test_size=0.25, random_state=SEED)
    s = d["aa_seq"].astype(str).str.slice(stop=2000)
    return d.index[s.isin(tr)], d.index[s.isin(va)], d.index[s.isin(te)]


def pair_grouped(d):
    tr, tmp = group_split(d.index, d["pair"].to_numpy(), 0.30, SEED)
    va, te = group_split(tmp, d.loc[tmp, "pair"].to_numpy(), 0.50, SEED)
    return tr, va, te


def gene_disjoint(d, p=(0.70, 0.15, 0.15)):
    rng = np.random.default_rng(SEED)
    genes = np.array(sorted(set(d["hgene"]) | set(d["tgene"])))
    assign = dict(zip(genes, rng.choice(3, size=len(genes), p=p)))
    h = d["hgene"].map(assign)
    t = d["tgene"].map(assign)
    same = h == t
    return d.index[same & (h == 0)], d.index[same & (h == 1)], d.index[same & (h == 2)]


def leakage(train, test):
    tr_pairs = set(train["pair"])
    tr_genes = set(train["hgene"]) | set(train["tgene"])
    tr_seqs = set(train["aa_seq"].astype(str))
    return {
        "test exact seq in train": test["aa_seq"].astype(str).isin(tr_seqs).mean(),
        "test pair in train": test["pair"].isin(tr_pairs).mean(),
        "test both genes in train": (test["hgene"].isin(tr_genes) & test["tgene"].isin(tr_genes)).mean(),
        "test any gene in train": (test["hgene"].isin(tr_genes) | test["tgene"].isin(tr_genes)).mean(),
    }


def main():
    labelset = sys.argv[1] if len(sys.argv) > 1 else "as_published"
    d = load(labelset)
    lines = [f"# Splits: {labelset}", "",
             f"Labelled records: {len(d):,} (Pathogenic {d.label.sum():,}, {d.label.mean():.1%}); "
             f"unique pairs {d.pair.nunique():,}; unique genes {len(set(d.hgene) | set(d.tgene)):,}", "",
             "| split | train | val | test | kept | test % pathogenic | test seq in train | "
             "test pair in train | test both genes in train | test any gene in train |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, fn in [("random_row", random_row), ("random_seq", random_seq),
                     ("pair_grouped", pair_grouped), ("gene_disjoint", gene_disjoint)]:
        tr, va, te = fn(d)
        out = P / "data/splits" / labelset / name
        out.mkdir(parents=True, exist_ok=True)
        for part, idx in [("train", tr), ("val", va), ("test", te)]:
            d.loc[idx, ["row_id", "hgene", "tgene", "label"]].to_csv(out / f"{part}.csv", index=False)
        L = leakage(d.loc[tr], d.loc[te])
        kept = (len(tr) + len(va) + len(te)) / len(d)
        lines.append(f"| {name} | {len(tr):,} | {len(va):,} | {len(te):,} | {kept:.0%} | "
                     f"{d.loc[te, 'label'].mean():.1%} | {L['test exact seq in train']:.1%} | "
                     f"{L['test pair in train']:.1%} | {L['test both genes in train']:.1%} | "
                     f"{L['test any gene in train']:.1%} |")
    lines += ["", "Splits use seed 42. gene_disjoint drops fusions whose two partners fall in different splits."]
    text = "\n".join(lines) + "\n"
    (P / f"results/splits_{labelset}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
