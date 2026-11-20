"""Gene-name-only and gene-role baselines on every split of the rebuilt FusionPath dataset.

Models (logistic regression, class_weight='balanced', fixed seed):
  gene_name : one-hot of head gene and tail gene (no sequence information)
  gene_role : 8 binary flags per fusion (head/tail x oncogene, TSG, ClinGen, CIViC),
              the annotations FusionPath's labels are derived from
Metrics on each test set: AUROC, AUPRC (and positive rate for reference).
For pair_grouped, the test set is also stratified by how many partner genes were seen
in training: C1 = both, C2 = one, C3 = none (Park & Marcotte 2012).

Usage: python train_baselines.py [as_published|noncancer_fixed]
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

SEED = 42
P = F2P_ROOT
D = P / "data/fusionpath_rebuild"
M = D / "01_source_data/merged"
FILES = {
    "as_published": "merged_fusions.filtered.hg38.formatted.labeled.csv",
    "noncancer_fixed": "merged_fusions.filtered.hg38.formatted.noncancerfix.labeled.csv",
}
SPLITS = ["random_row", "random_seq", "pair_grouped", "gene_disjoint"]


def annotation_sets():
    clingen = pd.read_csv(D / "fusion_annotations/Clingen-Gene-Disease-Summary-2026-01-05.csv", skiprows=4)
    clingen = set(clingen["GENE SYMBOL"].dropna().astype(str).str.upper())
    civic = set()
    for item in pd.read_csv(D / "fusion_annotations/civic_fusions.csv", header=None)[0].dropna().astype(str):
        for part in item.split("::"):
            civic.add(part.strip().upper().replace("[", "").replace("]", ""))
    return clingen, civic


def load(labelset):
    d = pd.read_csv(M / FILES[labelset], low_memory=False,
                    usecols=["hgene", "tgene", "hgene_role", "tgene_role", "cancer_label"])
    d["row_id"] = np.arange(len(d))
    d = d.set_index("row_id")
    for c in ["hgene", "tgene"]:
        d[c] = d[c].fillna("?").astype(str).str.upper()
    for c in ["hgene_role", "tgene_role"]:
        d[c] = d[c].fillna("").astype(str).str.upper()
    clingen, civic = annotation_sets()
    feats = pd.DataFrame(index=d.index)
    for side, g, r in [("h", "hgene", "hgene_role"), ("t", "tgene", "tgene_role")]:
        feats[f"{side}_onc"] = d[r].str.contains("ONCOGENE").astype(int)
        feats[f"{side}_tsg"] = d[r].str.contains("TSG").astype(int)
        feats[f"{side}_clingen"] = d[g].isin(clingen).astype(int)
        feats[f"{side}_civic"] = d[g].isin(civic).astype(int)
    return d, feats


def fit_eval(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return p


def metrics(y, p):
    if len(np.unique(y)) < 2:
        return {"n": len(y), "pos": int(y.sum()), "AUROC": np.nan, "AUPRC": np.nan}
    return {"n": len(y), "pos": int(y.sum()), "AUROC": roc_auc_score(y, p), "AUPRC": average_precision_score(y, p)}


def main():
    labelset = sys.argv[1] if len(sys.argv) > 1 else "as_published"
    d, feats = load(labelset)
    rows, preds = [], []
    for split in SPLITS:
        sd = P / "data/splits" / labelset / split
        tr = pd.read_csv(sd / "train.csv")
        te = pd.read_csv(sd / "test.csv")
        ytr, yte = tr["label"].to_numpy(), te["label"].to_numpy()

        enc = OneHotEncoder(handle_unknown="ignore")
        Xtr_name = enc.fit_transform(d.loc[tr.row_id, ["hgene", "tgene"]])
        Xte_name = enc.transform(d.loc[te.row_id, ["hgene", "tgene"]])
        Xtr_role = sparse.csr_matrix(feats.loc[tr.row_id].to_numpy())
        Xte_role = sparse.csr_matrix(feats.loc[te.row_id].to_numpy())

        out = {"gene_name": fit_eval(Xtr_name, ytr, Xte_name, yte),
               "gene_role": fit_eval(Xtr_role, ytr, Xte_role, yte),
               "constant": np.full(len(yte), ytr.mean())}

        tr_genes = set(d.loc[tr.row_id, "hgene"]) | set(d.loc[tr.row_id, "tgene"])
        seen = (d.loc[te.row_id, "hgene"].isin(tr_genes).to_numpy().astype(int)
                + d.loc[te.row_id, "tgene"].isin(tr_genes).to_numpy().astype(int))
        strata = {"all": np.ones(len(yte), bool)}
        if split == "pair_grouped":
            strata.update({"C1 both genes seen": seen == 2, "C2 one gene seen": seen == 1,
                           "C3 no gene seen": seen == 0})
        for model, p in out.items():
            for sname, mask in strata.items():
                rows.append({"split": split, "subset": sname, "model": model, **metrics(yte[mask], p[mask])})
            preds.append(pd.DataFrame({"split": split, "model": model, "row_id": te.row_id,
                                       "label": yte, "score": p, "genes_seen": seen}))

    res = pd.DataFrame(rows)
    res.to_csv(P / f"results/baselines_{labelset}.csv", index=False)
    pd.concat(preds).to_csv(P / f"results/baseline_predictions_{labelset}.csv.gz", index=False)
    fmt = res.copy()
    for c in ["AUROC", "AUPRC"]:
        fmt[c] = fmt[c].map(lambda x: "n/a" if pd.isna(x) else f"{x:.3f}")
    text = (f"# Baselines: {labelset}\n\nLogistic regression, class_weight=balanced, seed {SEED}.\n\n"
            + fmt.to_markdown(index=False) + "\n")
    (P / f"results/baselines_{labelset}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
