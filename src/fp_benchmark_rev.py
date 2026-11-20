"""Revision: FusionPath-style benchmark with normalised gene symbols, matched gene assignments and all baselines.

Small classifiers on frozen embeddings; no pLM fine-tuning.

Changes from the first version (train_seq_models.py, train_baselines.py, make_splits.py):
  * gene symbols normalised to HGNC (MLL -> KMT2A etc.) before any split, so a gene cannot cross splits under two names;
  * genes held out over 10 assignments (seeds 1-10), the same assignments for every model;
  * predictions saved with row_id, hgene, tgene and pair, so gene-level confidence intervals can be computed;
  * extra baselines: 3-mer composition, MMseqs2 homology to labelled training sequences;
  * extra embeddings: FusOn-pLM at its native context (fuson_plm_native), ESM-2 3B, ProtT5-XL;
  * a FusionPath-style model: FusOn-pLM fusion embedding + ProtBERT embeddings of the wild-type head and tail proteins.
Outputs: results/fp_rev/{metrics.csv, predictions.csv.gz, splits_<labelset>.md, summary.md}
"""
from f2p_root import F2P_ROOT, MMSEQS  # paths: see src/f2p_root.py
import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize

P = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_labels_v2 import norm  # noqa: E402  HGNC normalisation
from train_baselines import annotation_sets  # noqa: E402
from train_seq_models import train_mlp  # noqa: E402

M = P / "data/fusionpath_rebuild/01_source_data/merged"
FILES = {"as_published": "merged_fusions.filtered.hg38.formatted.labeled.csv",
         "noncancer_fixed": "merged_fusions.filtered.hg38.formatted.noncancerfix.labeled.csv"}
OUT = P / "results/fp_rev"
SEED = 42


def seq_norm(s):
    return str(s).replace("*", "").upper()


def sha(s):
    return hashlib.sha1(s.encode()).hexdigest()


def load(labelset):
    d = pd.read_csv(M / FILES[labelset], low_memory=False,
                    usecols=["hgene", "tgene", "hgene_role", "tgene_role", "aa_seq", "cancer_label"])
    d["row_id"] = np.arange(len(d))
    d = d.set_index("row_id")
    d = d[d.cancer_label.isin(["Benign", "Pathogenic"]) & d.aa_seq.notna()].copy()
    d["label"] = (d.cancer_label == "Pathogenic").astype(int)
    d["raw_h"] = d.hgene.fillna("?").astype(str).str.upper()
    d["raw_t"] = d.tgene.fillna("?").astype(str).str.upper()
    d["hgene"] = d.raw_h.map(norm)
    d["tgene"] = d.raw_t.map(norm)
    d["pair"] = d.hgene + "::" + d.tgene
    d["seq"] = d.aa_seq.map(seq_norm)
    d["sha1"] = d.seq.map(sha)
    clingen, civic = annotation_sets()
    clingen, civic = {norm(g) for g in clingen}, {norm(g) for g in civic}
    for side, g, r in [("h", "hgene", "hgene_role"), ("t", "tgene", "tgene_role")]:
        role = d[r].fillna("").astype(str).str.upper()
        d[f"{side}_onc"] = role.str.contains("ONCOGENE").astype(int)
        d[f"{side}_tsg"] = role.str.contains("TSG").astype(int)
        d[f"{side}_clingen"] = d[g].isin(clingen).astype(int)
        d[f"{side}_civic"] = d[g].isin(civic).astype(int)
    return d


# ------------------------------------------------------------------ splits (same rules as make_splits.py)
def _tts(items, test_size, seed):
    items = np.asarray(items)
    perm = np.random.default_rng(seed).permutation(len(items))
    n = int(np.ceil(test_size * len(items)))
    return items[perm[n:]], items[perm[:n]]


def _group(index, groups, test_size, seed):
    _, test_groups = _tts(pd.unique(groups), test_size, seed)
    m = np.isin(groups, test_groups)
    return index[~m], index[m]


def splits(d, n_assign):
    s = {}
    tr, tmp = _tts(d.index, 0.30, SEED)
    va, te = _tts(tmp, 0.50, SEED)
    s["random_row"] = (tr, va, te)
    seq2k = d.seq.str.slice(stop=2000)
    trv, te = _tts(seq2k.unique(), 0.20, SEED)
    tr, va = _tts(trv, 0.25, SEED)
    s["random_seq"] = (d.index[seq2k.isin(tr)], d.index[seq2k.isin(va)], d.index[seq2k.isin(te)])
    tr, tmp = _group(d.index, d.pair.to_numpy(), 0.30, SEED)
    va, te = _group(tmp, d.loc[tmp, "pair"].to_numpy(), 0.50, SEED)
    s["pair_grouped"] = (tr, va, te)
    genes = np.array(sorted(set(d.hgene) | set(d.tgene)))
    for k in range(1, n_assign + 1):
        a = dict(zip(genes, np.random.default_rng(k).choice(3, size=len(genes), p=(0.70, 0.15, 0.15))))
        h, t = d.hgene.map(a), d.tgene.map(a)
        s[f"genes_held_out_{k}"] = tuple(d.index[(h == t) & (h == j)].to_numpy() for j in range(3))
    return s


def leakage_table(d, s):
    rows = []
    for name, (tr, va, te) in s.items():
        T, E = d.loc[tr], d.loc[te]
        tg = set(T.hgene) | set(T.tgene)
        rows.append({"split": name, "train": len(tr), "val": len(va), "test": len(te), "test_pos": int(E.label.sum()),
                     "test_seq_in_train": E.seq.isin(set(T.seq)).mean(), "test_pair_in_train": E.pair.isin(set(T.pair)).mean(),
                     "test_both_genes_in_train": (E.hgene.isin(tg) & E.tgene.isin(tg)).mean(),
                     "test_any_gene_in_train": (E.hgene.isin(tg) | E.tgene.isin(tg)).mean()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ features
def load_emb(tag):
    parts, lut = [], {}
    for t in [tag, tag + "_v2new"]:
        f = P / f"data/embeddings/{t}"
        if not (f / "emb.npy").exists():
            continue
        ix = pd.read_csv(f / "index.csv")
        E = np.load(f / "emb.npy", mmap_mode="r")
        keep = [(s, i) for s, i in zip(ix.seq_sha1, ix.seq_idx) if s not in lut]
        off = sum(len(p) for p in parts)
        parts.append(np.asarray(E[[i for _, i in keep]], dtype=np.float32))
        lut.update({s: off + j for j, (s, _) in enumerate(keep)})
    return np.concatenate(parts), lut


def wt_lookup():
    wt = pd.read_csv(P / "external/fusionpath/fusionpath/data/gene_wt_mane_sequences.csv").dropna()
    out = {}
    for g, s in zip(wt.gene.astype(str).str.upper(), wt.sequence):
        sh = sha(seq_norm(s))
        out.setdefault(g, sh)
        out.setdefault(norm(g), sh)
    return out


def fusionpath_like(d, E, lut, W, wlut, wt):
    """[fusion embedding | ProtBERT wild-type head | ProtBERT wild-type tail | missing flags]."""
    F = E[[lut[s] for s in d.sha1]]
    blocks = [F]
    for raw, nrm in [("raw_h", "hgene"), ("raw_t", "tgene")]:
        shs = [wt.get(r) or wt.get(n) for r, n in zip(d[raw], d[nrm])]
        X = np.zeros((len(d), W.shape[1]), np.float32)
        miss = np.ones((len(d), 1), np.float32)
        for i, s in enumerate(shs):
            if s is not None and s in wlut:
                X[i] = W[wlut[s]]; miss[i] = 0
        blocks += [X, miss]
    return np.hstack(blocks)


def kmer_matrix(seqs):
    v = CountVectorizer(analyzer="char", ngram_range=(3, 3), lowercase=False)
    return normalize(v.fit_transform(seqs).astype(np.float32), norm="l1"), v


def homology_scores(train_seqs, train_labels, test_seqs, threads):
    """Best MMseqs2 bitscore to a labelled positive minus best to a negative (0 when no hit)."""
    with tempfile.TemporaryDirectory(dir=P / "tmp") as td:
        td = Path(td)
        tr = pd.DataFrame({"seq": train_seqs, "y": train_labels}).groupby("seq").y.max().reset_index()
        with open(td / "tr.fa", "w") as f:
            for i, (s, y) in enumerate(zip(tr.seq, tr.y)):
                f.write(f">t{i}_{y}\n{s}\n")
        uq = pd.unique(np.asarray(test_seqs))
        with open(td / "te.fa", "w") as f:
            for i, s in enumerate(uq):
                f.write(f">q{i}\n{s}\n")
        subprocess.run([str(MMSEQS), "easy-search", str(td / "te.fa"), str(td / "tr.fa"), str(td / "hits.m8"), str(td / "mm"),
                        "--threads", str(threads), "-s", "5.7", "--max-seqs", "300", "-v", "1",
                        "--format-output", "query,target,bits"], check=True)
        h = pd.read_csv(td / "hits.m8", sep="\t", names=["q", "t", "bits"])
        h["y"] = h.t.str.rsplit("_", n=1).str[1].astype(int)
        best = h.groupby(["q", "y"]).bits.max().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
        sc = (best[1] - best[0]).to_dict()
        qmap = {s: sc.get(f"q{i}", 0.0) for i, s in enumerate(uq)}
    return np.array([qmap[s] for s in test_seqs], dtype=np.float32)


def lr(Xtr, ytr, Xte, C=0.1, scale=True):
    if scale:
        sc = StandardScaler(with_mean=not sparse.issparse(Xtr)).fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    m = LogisticRegression(max_iter=5000, class_weight="balanced", C=C, random_state=SEED)
    return m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def metrics(y, p):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    return roc_auc_score(y, p), average_precision_score(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelsets", nargs="*", default=["as_published", "noncancer_fixed"])
    ap.add_argument("--assignments", type=int, default=10)
    ap.add_argument("--tags", nargs="*", default=["esm2_650m", "fuson_plm", "fuson_plm_native", "esm2_3b", "prot_t5"])
    ap.add_argument("--only-splits", nargs="*", default=None, help="exact split names, e.g. random_row genes_held_out_3")
    ap.add_argument("--run-name", default="all", help="suffix for output files, so chunks can run in parallel")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (P / "tmp").mkdir(exist_ok=True)
    t0 = time.time()
    emb = {t: load_emb(t) for t in a.tags}
    W, wlut = load_emb("protbert_wt")
    wt = wt_lookup()
    rows, preds = [], []
    for labelset in a.labelsets:
        d = load(labelset)
        ok = np.logical_and.reduce([d.sha1.isin(emb[t][1]).to_numpy() for t in a.tags])
        print(f"{labelset}: {len(d):,} labelled rows with a sequence; {int((~ok).sum())} lack an embedding (dropped); "
              f"{int((d.raw_h != d.hgene).sum() + (d.raw_t != d.tgene).sum())} symbols renamed by HGNC normalisation", flush=True)
        d = d[ok]
        S = splits(d, a.assignments)
        lk = leakage_table(d, S)
        (OUT / f"splits_{labelset}.md").write_text(f"# Splits ({labelset}, HGNC-normalised symbols)\n\n" + lk.round(4).to_markdown(index=False) + "\n")
        lk.to_csv(OUT / f"splits_{labelset}.csv", index=False)
        K, _ = kmer_matrix(d.seq.to_numpy())
        K = K.tocsr()
        pos = {rid: i for i, rid in enumerate(d.index)}
        role_cols = [f"{s}_{k}" for s in "ht" for k in ("onc", "tsg", "clingen", "civic")]
        main_models = labelset == "as_published"
        for name, (tr, va, te) in S.items():
            if a.only_splits and name not in a.only_splits:
                continue
            T, V, E = d.loc[tr], d.loc[va], d.loc[te]
            ytr, yva, yte = T.label.to_numpy(), V.label.to_numpy(), E.label.to_numpy()
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                continue
            out = {}
            enc = OneHotEncoder(handle_unknown="ignore").fit(T[["hgene", "tgene"]])
            out["gene_name"] = lr(enc.transform(T[["hgene", "tgene"]]), ytr, enc.transform(E[["hgene", "tgene"]]), C=1.0, scale=False)
            out["gene_role"] = lr(T[role_cols].to_numpy(float), ytr, E[role_cols].to_numpy(float), C=1.0, scale=False)
            ri = lambda X: K[[pos[r] for r in X.index]]  # noqa: E731
            out["kmer3"] = lr(ri(T), ytr, ri(E), C=1.0)
            if main_models:
                out["homology"] = homology_scores(T.seq.to_numpy(), ytr, E.seq.to_numpy(), a.threads)
            for tag in (a.tags if main_models else ["esm2_650m", "fuson_plm", "fuson_plm_native"]):
                Em, lut = emb[tag]
                X = lambda Z: Em[[lut[s] for s in Z.sha1]]  # noqa: E731
                sc = StandardScaler().fit(X(T))
                Xtr, Xva, Xte = [sc.transform(X(Z)).astype(np.float32) for Z in (T, V, E)]
                out[f"{tag}/logreg"] = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1,
                                                          random_state=SEED).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
                out[f"{tag}/mlp"] = np.mean([train_mlp(Xtr, ytr, Xva, yva, Xte, s) for s in (0, 1, 2)], axis=0)
            if main_models:
                Ef, lutf = emb["fuson_plm_native"]
                Z = [fusionpath_like(Q, Ef, lutf, W, wlut, wt) for Q in (T, V, E)]
                sc = StandardScaler().fit(Z[0])
                Ztr, Zva, Zte = [sc.transform(z).astype(np.float32) for z in Z]
                out["fusionpath_like/mlp"] = np.mean([train_mlp(Ztr, ytr, Zva, yva, Zte, s) for s in (0, 1, 2)], axis=0)
                out["fusionpath_like/logreg"] = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1,
                                                                   random_state=SEED).fit(Ztr, ytr).predict_proba(Zte)[:, 1]
            tg = set(T.hgene) | set(T.tgene)
            seen = E.hgene.isin(tg).astype(int).to_numpy() + E.tgene.isin(tg).astype(int).to_numpy()
            for model, p in out.items():
                au, ap_ = metrics(yte, p)
                rows.append({"labelset": labelset, "split": name, "model": model, "n_test": len(yte), "pos": int(yte.sum()),
                             "n_train": len(ytr), "AUROC": au, "AUPRC": ap_})
                preds.append(pd.DataFrame({"labelset": labelset, "split": name, "model": model, "row_id": E.index,
                                           "hgene": E.hgene.to_numpy(), "tgene": E.tgene.to_numpy(), "pair": E.pair.to_numpy(),
                                           "label": yte, "score": p, "genes_seen": seen}))
            print(f"  {labelset}/{name}: train {len(ytr):,}, test {len(yte):,} ({int(yte.sum())} pos), "
                  f"{len(out)} models, {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(OUT / f"metrics_{a.run_name}.csv", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"metrics_{a.run_name}.csv", index=False)
    pd.concat(preds).to_csv(OUT / f"predictions_{a.run_name}.csv.gz", index=False)
    res["split_group"] = res.split.str.replace(r"_\d+$", "", regex=True)
    summ = res.groupby(["labelset", "split_group", "model"]).agg(AUROC_mean=("AUROC", "mean"), AUROC_sd=("AUROC", "std"),
                                                                 AUPRC_mean=("AUPRC", "mean"), n_splits=("AUROC", "size"),
                                                                 test_pos_median=("pos", "median")).reset_index()
    (OUT / f"summary_{a.run_name}.md").write_text("# FusionPath-style benchmark, revision (HGNC-normalised symbols)\n\n"
                                    "genes_held_out = mean over 10 gene assignments (seeds 1-10).\n\n"
                                    + summ.round(3).to_markdown(index=False) + "\n")
    print((OUT / f"summary_{a.run_name}.md").read_text())
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
