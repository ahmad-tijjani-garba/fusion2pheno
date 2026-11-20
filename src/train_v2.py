"""Strict label set v2: shortcut baselines vs frozen protein-language-model embeddings, gene-held-out.

Small classifiers on frozen embeddings only; the pLMs are not fine-tuned.

Data      : data/labels_v2/strict/sequences.csv.gz (labelled, panel pairs removed, sequences with conflicting labels
            dropped), embeddings data/embeddings/{tag,tag_v2new}.
Split     : per seed, every gene goes to train/val/test with p = 0.70/0.15/0.15 (same rule as label_audit_v2 sec. 4).
            train = both genes train; val = both genes in train|val, at least one val; test = at least one gene test.
            A sequence that also occurs in a test pair is removed from train/val.
Unit      : models are fitted on sequences, weighted so every pair counts once and both classes weigh the same;
            a pair's score is the mean over its sequences; metrics are pair-level.
Subsets   : AB_all (pipeline + database sequences), AB_nonRT (read-through pairs removed), A_only (pipeline only).
Models    : length, geometry (read-through, inter-chromosomal, gene gap), role (FusionPath oncogene/TSG partner),
            gene_name (one-hot of genes seen in training), shortcuts (all four), <pLM>_logreg, <pLM>_mlp,
            <pLM>+shortcuts (logreg). Logreg C picked on validation pair AUROC.
Outputs   : results/v2_strict/{metrics_per_seed.csv, predictions.csv.gz, summary.md}
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

P = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_labels_v2 import norm  # noqa: E402

OUT = P / "results/v2_strict"
# revision: more embeddings (FusOn-pLM at native context, ESM-2 3B, ProtT5); override with F2P_TAGS="key=tag,..."
TAGS = dict(kv.split("=") for kv in os.environ.get("F2P_TAGS", "esm2=esm2_650m,fuson=fuson_plm,fuson_native=fuson_plm_native,"
                                                    "esm2_3b=esm2_3b,prot_t5=prot_t5").split(","))
SPLIT_MODE = os.environ.get("F2P_SPLIT", "gene")  # "gene" (default) or "family" (HGNC gene groups held out together)
KM = None  # 3-mer composition matrix aligned to the rows of s (set in load)
SEEDS = [42, *range(1, 11)]
CS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ data
def load():
    s = pd.read_csv(P / "data/labels_v2/strict/sequences.csv.gz", low_memory=False,
                    usecols=["pair", "set", "sha1", "label", "readthrough", "gene_gap_bp", "panel", "hgene", "tgene"])
    s = s[s.label.isin([0, 1]) & ~s.panel.astype(bool)]
    conflict = s.groupby("sha1").label.nunique()
    s = s[~s.sha1.isin(conflict.index[conflict > 1])]
    s = s.drop_duplicates(["pair", "sha1"]).reset_index(drop=True)
    s["label"] = s.label.astype(int)
    s["rt"] = (s.readthrough.astype(str) == "True").astype(float)
    gap = pd.to_numeric(s.gene_gap_bp, errors="coerce")
    s["geo_missing"] = gap.isna().astype(float)
    s["inter"] = np.isinf(gap).astype(float)
    s["log_gap"] = np.where(np.isfinite(gap), np.log10(gap.clip(lower=0) + 1), 0.0)

    roles = pd.read_csv(P / "data/fusionpath_rebuild/01_source_data/merged/merged_fusions.filtered.hg38.formatted.labeled.csv",
                        usecols=["hgene", "tgene", "hgene_role", "tgene_role"], low_memory=False)
    role = {norm(g) for g in pd.concat([roles.hgene[roles.hgene_role.notna()], roles.tgene[roles.tgene_role.notna()]])}
    s["role_h"] = s.hgene.isin(role).astype(float)
    s["role_t"] = s.tgene.isin(role).astype(float)

    lengths = {}
    for t in ["esm2_650m", "esm2_650m_v2new"]:
        ix = pd.read_csv(P / f"data/embeddings/{t}/index.csv")
        lengths.update(zip(ix.seq_sha1, ix.length))
    s["length"] = s.sha1.map(lengths)
    emb = {}
    for k, tag in TAGS.items():
        parts, ids = [], {}
        for t in [tag, tag + "_v2new"]:
            if not (P / f"data/embeddings/{t}/emb.npy").exists():
                continue
            ix = pd.read_csv(P / f"data/embeddings/{t}/index.csv")
            e = np.load(P / f"data/embeddings/{t}/emb.npy", mmap_mode="r")
            keep = [(sh, i) for sh, i in zip(ix.seq_sha1, ix.seq_idx) if sh not in ids]
            off = sum(len(p) for p in parts)
            sel = np.array([i for _, i in keep], dtype=int)
            parts.append(np.asarray(e[sel], dtype=np.float32))
            ids.update({sh: off + j for j, (sh, _) in enumerate(keep)})
        E = np.concatenate(parts)
        row = s.sha1.map(ids)
        assert row.notna().all(), f"{tag}: {row.isna().sum()} sequences without an embedding"
        emb[k] = E[row.astype(int).to_numpy()]
    assert s.length.notna().all()
    s["log_len"] = np.log10(s.length.astype(float))
    # revision: 3-mer composition baseline (sequence-only, no pLM)
    global KM
    import hashlib
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import normalize
    from embed_sequences import labelled_sequences
    seqs = {hashlib.sha1(x.encode()).hexdigest(): x for x in labelled_sequences()}
    for line in open(P / "data/labels_v2/strict/new_sequences.fasta"):
        if line.strip() and not line.startswith(">"):
            x = line.strip().replace("*", "").upper()
            seqs[hashlib.sha1(x.encode()).hexdigest()] = x
    sq = s.sha1.map(seqs)
    assert sq.notna().all(), f"{sq.isna().sum()} sequences missing for the k-mer baseline"
    KM = normalize(CountVectorizer(analyzer="char", ngram_range=(3, 3), lowercase=False).fit_transform(sq).astype(np.float32),
                   norm="l1").toarray()
    return s, emb


FAMILY = {}


def family_of(g):
    """HGNC gene group (first listed) of a gene; genes without a group form their own unit."""
    if not FAMILY:
        h = pd.read_csv(P / "data/labels_external/hgnc/hgnc_complete_set.txt", sep="\t", dtype=str,
                        usecols=["symbol", "gene_group_id"])
        FAMILY.update({sy.upper(): "fam" + gg.split("|")[0] for sy, gg in zip(h.symbol, h.gene_group_id)
                       if isinstance(gg, str) and gg})
    return FAMILY.get(g, g)


def split(pairs, seed, p=(0.70, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    unit = family_of if SPLIT_MODE == "family" else (lambda g: g)
    allg = set(pairs.hgene) | set(pairs.tgene)
    units = np.array(sorted({unit(g) for g in allg}))
    a0 = dict(zip(units, rng.choice(3, size=len(units), p=p)))
    a = {g: a0[unit(g)] for g in allg}
    h, t = pairs.hgene.map(a).to_numpy(), pairs.tgene.map(a).to_numpy()
    part = np.where((h == 2) | (t == 2), "test", np.where((h == 0) & (t == 0), "train", "val"))
    both = (h == 2) & (t == 2)
    return dict(zip(pairs.pair, part)), dict(zip(pairs.pair, both))


def weights(d):
    w = 1.0 / d.groupby("pair").pair.transform("size").to_numpy()
    for c in (0, 1):
        m = d.label.to_numpy() == c
        w[m] = w[m] / w[m].sum()
    return w * len(w) / w.sum()


# ------------------------------------------------------------------ features
def dense(d, cols):
    return d[cols].to_numpy(dtype=np.float32)


LEN = ["log_len"]
GEO = ["rt", "inter", "log_gap", "geo_missing"]
ROLE = ["role_h", "role_t"]


def gene_onehot(d, vocab):
    X = np.zeros((len(d), len(vocab)), dtype=np.float32)
    for r, (h, t) in enumerate(zip(d.hgene, d.tgene)):
        for g in (h, t):
            j = vocab.get(g)
            if j is not None:
                X[r, j] = 1.0
    return X


def features(name, d, vocab, emb_rows):
    blocks = []
    if name in ("length", "shortcuts", "plm+shortcuts"):
        blocks.append(dense(d, LEN))
        blocks.append(dense(d, LEN) ** 2)
    if name in ("geometry", "shortcuts", "plm+shortcuts"):
        blocks.append(dense(d, GEO))
    if name in ("role", "shortcuts", "plm+shortcuts"):
        blocks.append(dense(d, ROLE))
    if name in ("gene_name", "shortcuts", "plm+shortcuts"):
        blocks.append(gene_onehot(d, vocab))
    if name in ("plm", "plm+shortcuts"):
        blocks.append(emb_rows)
    return np.hstack(blocks)


# ------------------------------------------------------------------ models
def pair_scores(d, p):
    return pd.DataFrame({"pair": d.pair.to_numpy(), "label": d.label.to_numpy(), "p": p}).groupby("pair").agg(
        label=("label", "first"), p=("p", "mean"))


def pair_auc(d, p):
    g = pair_scores(d, p)
    return roc_auc_score(g.label, g.p) if g.label.nunique() == 2 else np.nan


def fit_logreg(Xtr, ytr, wtr, Xva, dva, Xte):
    sc = StandardScaler().fit(Xtr, sample_weight=wtr)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    best = (-1, None)
    for C in CS:
        m = LogisticRegression(C=C, max_iter=5000).fit(Xtr, ytr, sample_weight=wtr)
        a = pair_auc(dva, m.predict_proba(Xva)[:, 1])
        if a > best[0]:
            best = (a, C)
    m = LogisticRegression(C=best[1], max_iter=5000).fit(Xtr, ytr, sample_weight=wtr)
    return m.predict_proba(Xte)[:, 1], best[1]


def fit_mlp(Xtr, ytr, wtr, Xva, dva, Xte, seed):
    sc = StandardScaler().fit(Xtr, sample_weight=wtr)
    T = lambda x: torch.tensor(sc.transform(x), dtype=torch.float32, device=DEV)  # noqa: E731
    Xtr_t, Xva_t, Xte_t = T(Xtr), T(Xva), T(Xte)
    y_t = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    w_t = torch.tensor(wtr, dtype=torch.float32, device=DEV)
    preds = []
    for init in range(3):
        torch.manual_seed(1000 * seed + init)
        net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                                  torch.nn.Linear(256, 1)).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        g = torch.Generator().manual_seed(1000 * seed + init)
        best, state, patience = -1, None, 0
        for epoch in range(100):
            net.train()
            perm = torch.randperm(len(Xtr_t), generator=g).to(DEV)
            for i in range(0, len(perm), 128):
                b = perm[i:i + 128]
                opt.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xtr_t[b]).squeeze(1), y_t[b], reduction="none")
                ((loss * w_t[b]).sum() / w_t[b].sum()).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                a = pair_auc(dva, torch.sigmoid(net(Xva_t).squeeze(1)).cpu().numpy())
            if a > best:
                best, patience = a, 0
                state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                patience += 1
                if patience >= 10:
                    break
        net.load_state_dict(state)
        net.eval()
        with torch.no_grad():
            preds.append(torch.sigmoid(net(Xte_t).squeeze(1)).cpu().numpy())
    return np.mean(preds, axis=0)


# ------------------------------------------------------------------ run
def run(s, emb, subsets, seeds, mlp=True):
    allpairs = s.drop_duplicates("pair")[["pair", "hgene", "tgene"]]
    preds, rows = [], []
    for sub in subsets:
        d0 = s if sub == "AB_all" else s[s.rt == 0] if sub == "AB_nonRT" else s[s.set == "A_pipeline"]
        for seed in seeds:
            t0 = time.time()
            part, both = split(allpairs, seed)
            d = d0.assign(part=d0.pair.map(part))
            test_seqs = set(d.sha1[d.part == "test"])
            tr = d[(d.part == "train") & ~d.sha1.isin(test_seqs)]
            va = d[(d.part == "val") & ~d.sha1.isin(test_seqs)]
            te = d[d.part == "test"]
            if min(tr.label.nunique(), va.label.nunique(), te.label.nunique()) < 2:
                print(f"{sub} seed {seed}: a fold lacks a class, skipped", flush=True)
                continue
            vocab = {g: j for j, g in enumerate(sorted(set(tr.hgene) | set(tr.tgene)))}
            ytr, wtr = tr.label.to_numpy(), weights(tr)
            idx = {k: (tr.index.to_numpy(), va.index.to_numpy(), te.index.to_numpy()) for k in emb}
            jobs = [(m, None) for m in ["length", "geometry", "role", "gene_name", "shortcuts"]]
            jobs += [(m, k) for k in emb for m in ["plm", "plm+shortcuts"]]
            for m, k in jobs:
                E = emb[k] if k else None
                F = [features(m, x, vocab, E[i] if k else None) for x, i in zip((tr, va, te), idx[k] if k else (None,) * 3)]
                p, C = fit_logreg(F[0], ytr, wtr, F[1], va, F[2])
                name = m if k is None else m.replace("plm", k) + ("_logreg" if m == "plm" else "")
                preds.append(pair_scores(te, p).assign(subset=sub, seed=seed, model=name, C=C))
            if KM is not None:  # revision: 3-mer composition baseline
                p, C = fit_logreg(KM[tr.index.to_numpy()], ytr, wtr, KM[va.index.to_numpy()], va, KM[te.index.to_numpy()])
                preds.append(pair_scores(te, p).assign(subset=sub, seed=seed, model="kmer3", C=C))
            if mlp:
                for k in emb:
                    i = idx[k]
                    p = fit_mlp(emb[k][i[0]], ytr, wtr, emb[k][i[1]], va, emb[k][i[2]], seed)
                    preds.append(pair_scores(te, p).assign(subset=sub, seed=seed, model=f"{k}_mlp", C=np.nan))
            npos = te.drop_duplicates("pair").label.sum()
            print(f"{sub} seed {seed}: train {tr.pair.nunique()} pairs / {len(tr)} seqs, val {va.pair.nunique()}, "
                  f"test {te.pair.nunique()} pairs ({npos} pos), {time.time() - t0:.0f}s", flush=True)
    return pd.concat(preds).reset_index()


def mark_both_unseen(pr, s):
    allpairs = s.drop_duplicates("pair")[["pair", "hgene", "tgene"]]
    out = []
    for seed, g in pr.groupby("seed"):
        both = split(allpairs, seed)[1]
        out.append(g.assign(both_unseen=g.pair.map(both)))
    return pd.concat(out)


def per_seed(pr):
    rows = []
    for (sub, seed, model), g in pr.groupby(["subset", "seed", "model"]):
        rows.append({"subset": sub, "seed": seed, "model": model, "n_pairs": len(g), "n_pos": int(g.label.sum()),
                     "auroc": roc_auc_score(g.label, g.p), "auprc": average_precision_score(g.label, g.p),
                     "prevalence": g.label.mean()})
    return pd.DataFrame(rows)


def fast_auc(y, p):
    r = pd.Series(p).rank().to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bootstrap(pr, sub, models, B=1000, rng_seed=0):
    """Mean-over-seeds AUROC per model, and paired differences, with pair-bootstrap 95% CIs (resampled within seed)."""
    rng = np.random.default_rng(rng_seed)
    q = pr[pr.subset == sub]
    seeds = sorted(q.seed.unique())
    wide = {sd: q[q.seed == sd].pivot_table(index="pair", columns="model", values="p").join(
        q[q.seed == sd].groupby("pair").label.first()) for sd in seeds}
    boot = np.zeros((B, len(models)))
    point = np.zeros(len(models))
    for sd in seeds:
        w = wide[sd]
        y = w.label.to_numpy()
        P_ = w[models].to_numpy()
        point += np.array([fast_auc(y, P_[:, j]) for j in range(len(models))])
        n = len(y)
        for b in range(B):
            i = rng.integers(0, n, n)
            while y[i].min() == y[i].max():
                i = rng.integers(0, n, n)
            boot[b] += [fast_auc(y[i], P_[i, j]) for j in range(len(models))]
    return point / len(seeds), boot / len(seeds)


def summary(pr, ps):
    L = ["# Strict label set v2: shortcut baselines vs protein-language-model embeddings\n",
         "Gene-held-out (at least one gene of every test pair unseen in training), 11 gene assignments, pair-level metrics. "
         "AUROC: mean over assignments with 95% pair-bootstrap CI (resampled within each assignment). "
         "sd = spread across assignments.\n"]
    base = ["length", "geometry", "role", "gene_name", "shortcuts"]
    for sub in ps.subset.unique():
        q = ps[ps.subset == sub]
        models = [m for m in base if m in set(q.model)] + sorted(m for m in set(q.model) if m not in base)
        point, boot = bootstrap(pr, sub, models)
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        agg = q.groupby("model").agg(auroc_sd=("auroc", "std"), auprc=("auprc", "mean"), prevalence=("prevalence", "mean"),
                                     test_pairs=("n_pairs", "median"), test_pos=("n_pos", "median"))
        t = pd.DataFrame({"AUROC": point, "95% CI": [f"{a:.3f}-{b:.3f}" for a, b in zip(lo, hi)]}, index=models).join(agg)
        L.append(f"## {sub}\n")
        L.append(t.round(3).to_markdown() + "\n")
        j = {m: i for i, m in enumerate(models)}
        D = []
        for a, b in [("esm2_logreg", "shortcuts"), ("fuson_logreg", "shortcuts"), ("esm2_mlp", "shortcuts"),
                     ("fuson_mlp", "shortcuts"), ("esm2+shortcuts", "shortcuts"), ("fuson+shortcuts", "shortcuts"),
                     ("esm2_logreg", "length"), ("fuson_logreg", "length"), ("fuson_logreg", "esm2_logreg"),
                     ("fuson+shortcuts", "esm2+shortcuts")]:
            if a in j and b in j:
                diff = boot[:, j[a]] - boot[:, j[b]]
                D.append({"comparison": f"{a} - {b}", "dAUROC": point[j[a]] - point[j[b]],
                          "95% CI": "{:.3f} to {:.3f}".format(*np.percentile(diff, [2.5, 97.5])),
                          "P(boot <= 0)": (diff <= 0).mean(),
                          "seeds where a > b": int((q.pivot(index="seed", columns="model", values="auroc").eval(f"`{a}` > `{b}`")).sum())})
        L.append("Paired differences:\n")
        L.append(pd.DataFrame(D).round(3).to_markdown(index=False) + "\n")
        bu = pr[(pr.subset == sub) & pr.both_unseen]
        if len(bu):
            r = []
            for m in models:
                g = bu[bu.model == m]
                if g.label.nunique() == 2:
                    r.append({"model": m, "pooled pairs": len(g), "pos": int(g.label.sum()), "AUROC (pooled, exploratory)": roc_auc_score(g.label, g.p)})
            L.append("Both genes unseen (pooled over assignments, the same pair can recur; exploratory only):\n")
            L.append(pd.DataFrame(r).round(3).to_markdown(index=False) + "\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--subsets", nargs="*", default=["AB_all", "AB_nonRT", "A_only"])
    ap.add_argument("--no-mlp", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    s, emb = load()
    print(f"loaded {len(s):,} (pair, sequence) rows, {s.pair.nunique()} pairs, {s.sha1.nunique()} sequences, "
          f"device {DEV}, {time.time() - t0:.0f}s", flush=True)
    pr = mark_both_unseen(run(s, emb, a.subsets, a.seeds, mlp=not a.no_mlp), s)
    pr.to_csv(out / "predictions.csv.gz", index=False)
    ps = per_seed(pr)
    ps.to_csv(out / "metrics_per_seed.csv", index=False)
    md = summary(pr, ps)
    (out / "summary.md").write_text(md)
    print(md)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
