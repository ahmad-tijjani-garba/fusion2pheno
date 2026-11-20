"""Train sequence-only classifiers on frozen pLM embeddings and evaluate on every split.

Inputs: data/embeddings/<tag>/{emb.npy,index.csv}; data/splits/<labelset>/<split>/*.csv
Models per (labelset, split, embedding):
  logreg : standardised features, LogisticRegression(class_weight=balanced)
  mlp    : 1280 -> 256 -> 1 (ReLU, dropout 0.2), AdamW, pos-weighted BCE,
           early stopping on validation AUPRC, 3 seeds (mean reported)
Extra: gene_disjoint repeated with 5 additional gene-assignment seeds (gd_s1..gd_s5),
because one gene-disjoint test set is small.
Outputs: results/seq_models.csv, results/seq_models.md, results/seq_model_predictions.csv.gz
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

P = F2P_ROOT
M = P / "data/fusionpath_rebuild/01_source_data/merged"
FILES = {"as_published": "merged_fusions.filtered.hg38.formatted.labeled.csv",
         "noncancer_fixed": "merged_fusions.filtered.hg38.formatted.noncancerfix.labeled.csv"}
TAGS = ["esm2_650m", "fuson_plm"]
SPLITS = ["random_row", "random_seq", "pair_grouped", "gene_disjoint"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def norm_seq(s):
    return str(s).replace("*", "").upper()


def load_embeddings():
    out = {}
    for t in TAGS:
        idx = pd.read_csv(P / f"data/embeddings/{t}/index.csv")
        out[t] = (np.load(P / f"data/embeddings/{t}/emb.npy").astype(np.float32),
                  dict(zip(idx.seq_sha1, idx.seq_idx)))
    return out


def load_rows(labelset):
    d = pd.read_csv(M / FILES[labelset], low_memory=False,
                    usecols=["hgene", "tgene", "aa_seq", "cancer_label"])
    d["row_id"] = np.arange(len(d))
    d = d.set_index("row_id")
    d = d[d.cancer_label.isin(["Benign", "Pathogenic"])].copy()
    d["label"] = (d.cancer_label == "Pathogenic").astype(int)
    d["sha1"] = [hashlib.sha1(norm_seq(s).encode()).hexdigest() if pd.notna(s) else None for s in d.aa_seq]
    for c in ["hgene", "tgene"]:
        d[c] = d[c].fillna("?").astype(str).str.upper()
    # revision: HGNC-normalised symbols (MLL -> KMT2A etc.) so a gene cannot cross splits under two names
    from build_labels_v2 import norm
    d["hgene"] = d.hgene.map(norm)
    d["tgene"] = d.tgene.map(norm)
    return d


def gene_disjoint_ids(d, seed, p=(0.70, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    genes = np.array(sorted(set(d.hgene) | set(d.tgene)))
    a = dict(zip(genes, rng.choice(3, size=len(genes), p=p)))
    h, t = d.hgene.map(a), d.tgene.map(a)
    same = h == t
    return [d.index[same & (h == k)].to_numpy() for k in range(3)]


def metrics(y, p):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    return roc_auc_score(y, p), average_precision_score(y, p)


def train_mlp(Xtr, ytr, Xva, yva, Xte, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 256), torch.nn.ReLU(),
                              torch.nn.Dropout(0.2), torch.nn.Linear(256, 1)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pw = torch.tensor([(1 - ytr.mean()) / max(ytr.mean(), 1e-6)], device=DEV)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    Xtr_t, ytr_t = torch.tensor(Xtr, device=DEV), torch.tensor(ytr, dtype=torch.float32, device=DEV)
    Xva_t, Xte_t = torch.tensor(Xva, device=DEV), torch.tensor(Xte, device=DEV)
    best, best_state, patience = -1, None, 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(60):
        net.train()
        perm = torch.randperm(len(Xtr_t), generator=g).to(DEV)
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            lossf(net(Xtr_t[b]).squeeze(1), ytr_t[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pva = torch.sigmoid(net(Xva_t).squeeze(1)).cpu().numpy()
        score = average_precision_score(yva, pva) if len(np.unique(yva)) > 1 else 0
        if score > best:
            best, patience = score, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 8:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(Xte_t).squeeze(1)).cpu().numpy()


def main():
    t0 = time.time()
    emb = load_embeddings()
    rows, preds = [], []
    print(f"device: {DEV} {torch.cuda.get_device_name(0) if DEV == 'cuda' else ''}", flush=True)
    for labelset in FILES:
        d = load_rows(labelset)
        missing = int((~d.sha1.isin(emb[TAGS[0]][1].keys())).sum())
        print(f"{labelset}: {len(d):,} labelled rows, {missing} without embedding (dropped)", flush=True)
        d = d[d.sha1.isin(emb[TAGS[0]][1].keys())]

        split_sets = {}
        for s in SPLITS:
            sd = P / "data/splits" / labelset / s
            split_sets[s] = [pd.read_csv(sd / f"{x}.csv").row_id.to_numpy() for x in ["train", "val", "test"]]
        for k in range(1, 6):
            split_sets[f"gd_s{k}"] = gene_disjoint_ids(d, seed=k)

        for split, (tr, va, te) in split_sets.items():
            tr, va, te = [np.intersect1d(x, d.index.to_numpy()) for x in (tr, va, te)]
            ytr, yva, yte = d.loc[tr, "label"].to_numpy(), d.loc[va, "label"].to_numpy(), d.loc[te, "label"].to_numpy()
            tr_genes = set(d.loc[tr, "hgene"]) | set(d.loc[tr, "tgene"])
            seen = (d.loc[te, "hgene"].isin(tr_genes).to_numpy().astype(int)
                    + d.loc[te, "tgene"].isin(tr_genes).to_numpy().astype(int))
            strata = {"all": np.ones(len(te), bool)}
            if split == "pair_grouped":
                strata.update({"C1": seen == 2, "C2": seen == 1, "C3": seen == 0})
            for tag in TAGS:
                E, lut = emb[tag]
                X = lambda ids: E[[lut[h] for h in d.loc[ids, "sha1"]]]
                sc = StandardScaler().fit(X(tr))
                Xtr, Xva, Xte = [sc.transform(X(i)).astype(np.float32) for i in (tr, va, te)]
                lr = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1, random_state=42)
                p_lr = lr.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
                p_mlp = np.mean([train_mlp(Xtr, ytr, Xva, yva, Xte, seed) for seed in (0, 1, 2)], axis=0)
                for model, p in [("logreg", p_lr), ("mlp", p_mlp)]:
                    for sname, mask in strata.items():
                        au, ap = metrics(yte[mask], p[mask])
                        rows.append({"labelset": labelset, "split": split, "subset": sname, "embedding": tag,
                                     "model": model, "n_test": int(mask.sum()), "pos": int(yte[mask].sum()),
                                     "AUROC": au, "AUPRC": ap})
                    preds.append(pd.DataFrame({"labelset": labelset, "split": split, "embedding": tag,
                                               "model": model, "row_id": te, "label": yte, "score": p,
                                               "genes_seen": seen}))
            print(f"  {labelset}/{split} done ({time.time() - t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(P / "results/seq_models.csv", index=False)
    pd.concat(preds).to_csv(P / "results/seq_model_predictions.csv.gz", index=False)

    # summary: main splits + gene-disjoint mean/sd over seeds 42,1..5
    main_tab = res[(res.subset == "all") & res.split.isin(SPLITS)]
    gd = res[(res.subset == "all") & res.split.str.match(r"^(gene_disjoint|gd_s\d)$")]
    gd_sum = gd.groupby(["labelset", "embedding", "model"]).agg(
        AUROC_mean=("AUROC", "mean"), AUROC_sd=("AUROC", "std"),
        AUPRC_mean=("AUPRC", "mean"), AUPRC_sd=("AUPRC", "std"), n_splits=("AUROC", "size")).reset_index()
    strat = res[res.subset.isin(["C1", "C2", "C3"])]
    text = ["# Sequence-only classifiers on frozen pLM embeddings", "",
            "## Main splits (test, all records)", "", main_tab.round(3).to_markdown(index=False), "",
            "## Gene-disjoint over 6 gene assignments (seed 42 + 1..5)", "", gd_sum.round(3).to_markdown(index=False), "",
            "## pair_grouped test by genes seen in training (C1 both, C2 one, C3 none)", "",
            strat.round(3).to_markdown(index=False), ""]
    (P / "results/seq_models.md").write_text("\n".join(text))
    print("\n".join(text))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
