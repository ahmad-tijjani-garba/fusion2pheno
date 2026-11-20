"""Round-2 review analyses. Logistic regression only; CPU.

1. label sensitivity: gene-role-free labels with stricter positives
   tier1_only    positives need CIViC level A-C or ChimerKB (drops Mitelman-only positives)
   mitelman_ge3  Mitelman-only positives need >= 3 cases in focused references (instead of >= 2)
   Same design as train_v2.py (11 gene assignments; AB_all and AB_nonRT; LR with C chosen on validation).
2. missing-partner flags: LR on FusionPath's two 'no wild-type sequence' flags alone, on the 13 FusionPath-style splits.
3. paralogue check: which negative pairs make up the both-genes-unseen, non-read-through subset (50 assignments),
   their HGNC gene groups, and the embedding gain with the most frequent negative group removed.
Outputs: results/round2/{sensitivity_predictions.csv.gz, sensitivity.csv, flags.csv, paralogues.csv, paralogue_gain.csv, summary.md}
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

P = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reanalysis as RA  # noqa: E402
import train_v2 as T  # noqa: E402

OUT = P / "results/round2"
OUT.mkdir(parents=True, exist_ok=True)
md = ["# Round-2 review analyses\n"]


def boot(pr, genes, models, comps, extra, filt=None):
    rows = []
    q = pr if filt is None else pr[filt(pr)]
    wide = q.pivot_table(index=["seed", "pair", "hgene", "tgene", "label"], columns="model", values="p").reset_index()
    blocks = [b.dropna(subset=models) for _, b in wide.groupby("seed")]
    blocks = [b for b in blocks if 0 < b.label.sum() < len(b)]
    if not blocks:
        return rows
    pt, dr, per = RA.gene_boot(blocks, models, genes)
    return RA.summarise(pt, dr, per, models, comps, {**extra, "median_test_pairs": float(np.median([len(b) for b in blocks])),
                                                     "median_test_neg": float(np.median([(b.label == 0).sum() for b in blocks]))})


# ---------------------------------------------------------------- 1. label sensitivity
s, emb = T.load()
lab = pd.read_csv(P / "data/labels_v2/strict/sequences.csv.gz", usecols=["pair", "tier1", "positive_ge3"],
                  low_memory=False).drop_duplicates("pair").set_index("pair")
genes_of = dict(zip(s.pair, zip(s.hgene, s.tgene)))
genes = sorted(set(s.hgene) | set(s.tgene))
preds, rows = [], []
for variant, col in [("tier1_only", "tier1"), ("mitelman_ge3", "positive_ge3")]:
    keep = (s.label == 0) | s.pair.map(lab[col]).fillna(False).astype(bool)
    sv = s[keep]  # original row index kept, so embedding rows stay aligned
    u = sv.drop_duplicates("pair")
    print(f"{variant}: {int((u.label == 1).sum())} positive / {int((u.label == 0).sum())} negative pairs", flush=True)
    md.append(f"- {variant}: {int((u.label == 1).sum())} positive and {int((u.label == 0).sum())} negative pairs")
    pr = T.mark_both_unseen(T.run(sv, emb, ["AB_all", "AB_nonRT"], T.SEEDS, mlp=False), sv)
    pr["variant"] = variant
    pr["hgene"] = pr.pair.map(lambda p: genes_of[p][0]); pr["tgene"] = pr.pair.map(lambda p: genes_of[p][1])
    preds.append(pr)
    tags = list(emb)
    models = ["shortcuts", "gene_name", "length", "geometry"] + [f"{k}_logreg" for k in tags] + [f"{k}+shortcuts" for k in tags] + ["kmer3"]
    comps = [(f"{k}+shortcuts", "shortcuts") for k in tags] + [(f"{k}_logreg", "shortcuts") for k in tags]
    for sub, q in pr.groupby("subset"):
        ms = [m for m in models if m in set(q.model)]
        rows += boot(q, genes, ms, comps, {"variant": variant, "subset": sub, "test_pairs": "all test pairs"})
        rows += boot(q, genes, ms, comps, {"variant": variant, "subset": sub, "test_pairs": "both genes unseen"},
                     filt=lambda z: z.both_unseen.astype(bool))
pd.concat(preds).to_csv(OUT / "sensitivity_predictions.csv.gz", index=False)
sens = pd.DataFrame(rows)
sens.to_csv(OUT / "sensitivity.csv", index=False)
md += ["", "## 1. Label sensitivity (gene-resampled 95% CIs, 11 assignments)", "",
       sens.round(3).to_markdown(index=False), ""]

# ---------------------------------------------------------------- 2. missing-partner flags (FusionPath-style labels)
B = np.load(P / "data/domains/fpr_bundle.npz", allow_pickle=True)
meta = pd.read_csv(P / "data/domains/fpr_meta.csv.gz")
S = json.load(open(P / "data/domains/fpr_splits.json"))
Z = B["Z"]
fl = np.stack([Z[:, 1280 + 1024], Z[:, 1280 + 1024 + 1 + 1024]], axis=1).astype(np.float32)
assert set(np.unique(fl)) <= {0.0, 1.0}, "flag columns are not binary"
y = B["label"]; pos = {r: i for i, r in enumerate(B["row_id"])}
md += ["## 2. Missing-partner flags", "",
       f"- share of records with no wild-type sequence: head {fl[:, 0].mean():.3f}, tail {fl[:, 1].mean():.3f}",
       f"- among pathogenic: head {fl[y == 1, 0].mean():.3f}, tail {fl[y == 1, 1].mean():.3f}; "
       f"among benign: head {fl[y == 0, 0].mean():.3f}, tail {fl[y == 0, 1].mean():.3f}"]
fp = []
for name, (tr, va, te) in S.items():
    itr, ite = [pos[r] for r in tr], [pos[r] for r in te]
    if len(np.unique(y[ite])) < 2:
        continue
    p = LogisticRegression(class_weight="balanced", max_iter=2000).fit(fl[itr], y[itr]).predict_proba(fl[ite])[:, 1]
    m = meta.iloc[ite]
    fp.append(pd.DataFrame({"labelset": "as_published", "split": name, "model": "missing_flags", "row_id": m.row_id.to_numpy(),
                            "hgene": m.hgene.to_numpy(), "tgene": m.tgene.to_numpy(), "pair": m.pair.to_numpy(),
                            "label": y[ite], "score": p}))
fp = pd.concat(fp)
old = pd.concat([pd.read_csv(f) for f in sorted((P / "results/fp_rev").glob("predictions_*.csv.gz"))] +
                [pd.read_csv(P / "results/fp_rev_fpr/predictions_FPR.csv.gz")])
old = old[(old.labelset == "as_published") & old.model.isin(["gene_name", "fusionpath_like/mlp", "fusionpath_rebuild/full"])]
q = pd.concat([old, fp])
fg = sorted(set(q.hgene) | set(q.tgene))
mods = ["missing_flags", "gene_name", "fusionpath_like/mlp", "fusionpath_rebuild/full"]
w = q.pivot_table(index=["split", "row_id", "hgene", "tgene", "label"], columns="model", values="score").reset_index()
frows = []
for g, sps in {"random_row": ["random_row"], "pair_grouped": ["pair_grouped"],
               "genes_held_out": [f"genes_held_out_{k}" for k in range(1, 11)]}.items():
    blocks = [w[w.split == sp].dropna(subset=mods) for sp in sps]
    blocks = [b for b in blocks if 0 < b.label.sum() < len(b)]
    pt, dr, per = RA.gene_boot(blocks, mods, fg)
    frows += RA.summarise(pt, dr, per, mods, [("missing_flags", "gene_name"), ("fusionpath_like/mlp", "missing_flags")],
                          {"split": g})
flags = pd.DataFrame(frows)
flags.to_csv(OUT / "flags.csv", index=False)
md += ["", flags.round(3).to_markdown(index=False), ""]

# ---------------------------------------------------------------- 3. paralogues among both-unseen non-read-through negatives
v = pd.concat([pd.read_csv(P / "results/v2_rev/seeds50/predictions.csv.gz")] +
              ([pd.read_csv(P / "results/v2_rev/seeds50/predictions_checks.csv.gz")]
               if (P / "results/v2_rev/seeds50/predictions_checks.csv.gz").exists() else []))
v = v[(v.subset == "AB_nonRT")].drop(columns=["both_unseen"], errors="ignore")
v = T.mark_both_unseen(v, s)
v = v[v.both_unseen.astype(bool)]
v["hgene"] = v.pair.map(lambda p: genes_of[p][0]); v["tgene"] = v.pair.map(lambda p: genes_of[p][1])
neg = v[(v.label == 0)].drop_duplicates(["seed", "pair"])
fam = lambda g: T.family_of(g)  # noqa: E731
neg = neg.assign(hfam=neg.hgene.map(fam), tfam=neg.tgene.map(fam))
neg["group"] = np.where(neg.hfam == neg.tfam, neg.hfam, neg.hfam + "|" + neg.tfam)
cnt = neg.groupby(["pair", "group"]).seed.nunique().rename("assignments").reset_index().sort_values("assignments", ascending=False)
cnt.to_csv(OUT / "paralogues.csv", index=False)
gsize = neg.groupby("group").pair.nunique().sort_values(ascending=False)
top = gsize.index[0]
md += ["## 3. Negatives in the both-genes-unseen, non-read-through subset (50 assignments)", "",
       f"- {neg.pair.nunique()} distinct negative pairs over {neg.seed.nunique()} assignments; median "
       f"{neg.groupby('seed').pair.nunique().median():.0f} per assignment",
       f"- largest HGNC group: {top} ({gsize.iloc[0]} pairs; {(neg.group == top).mean():.1%} of negative appearances)",
       "", cnt.head(25).to_markdown(index=False), ""]
mods3 = ["shortcuts", "esm2+shortcuts", "esm2_logreg", "fuson+shortcuts"]
mods3 = [m for m in mods3 if m in set(v.model)]
comp3 = [(m, "shortcuts") for m in mods3 if m != "shortcuts"]
g3 = sorted(set(s.hgene) | set(s.tgene))
r3 = boot(v, g3, mods3, comp3, {"set": "all both-unseen non-RT pairs"})
drop = set(neg.pair[neg.group == top])
r3 += boot(v[~v.pair.isin(drop)], g3, mods3, comp3, {"set": f"without negative pairs of {top}"})
pg = pd.DataFrame(r3)
pg.to_csv(OUT / "paralogue_gain.csv", index=False)
md += [pg.round(3).to_markdown(index=False), ""]
(OUT / "summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
