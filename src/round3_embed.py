"""Round-3 review items 3 and 4. Logistic regression only; CPU.

Item 3, a more sensitive test of what embeddings add to the shortcut features (gene-role-free labels):
  <k>_pca32+shortcuts  embedding reduced to 32 principal components (fitted on the training sequences) + shortcut features
  <k>_stack            late fusion: separate logistic regressions on the shortcut features and on the embedding, combined
                       by a two-input logistic regression fitted on the validation pairs (test pairs untouched)
  shortcuts            refitted in the same run, for paired comparisons
  Pooled sources AB_all and AB_nonRT (seeds 1-50) and the four single sources (seeds 42, 1-20); k = esm2, fuson.
Item 4, what frozen embeddings encode on held-out genes (AB_all, seeds 42 and 1-10): logistic-regression probes from
  ESM-2 or FusOn-pLM embeddings predicting long protein (above the median length), read-through, inter-chromosomal,
  any partner with an oncogene/tumour-suppressor role, our translation (vs database sequence), and the label itself.
  Scored per test sequence; gene-resampled 95% CIs.
Outputs in results/round3/: embed_add.csv, embed_probes.csv, round3_embed.md (+ predictions)
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

P = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reanalysis as RA  # noqa: E402
import train_v2 as T  # noqa: E402
import train_v2_checks as C  # noqa: E402
import train_v2_matched as M  # noqa: E402

QUICK = "--quick" in sys.argv
OUT = P / ("results/round3_quick" if QUICK else "results/round3")
OUT.mkdir(parents=True, exist_ok=True)
KEYS = ["esm2", "fuson"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_lr_val_test(Xtr, ytr, wtr, Xva, va, Xte):
    """fit_logreg with C chosen on validation; returns validation and test probabilities."""
    p, Cc = T.fit_logreg(Xtr, ytr, wtr, Xva, va, np.vstack([Xva, Xte]))
    return p[:len(Xva)], p[len(Xva):], Cc


def split_sets(d0, allpairs, seed):
    part, _ = T.split(allpairs, seed)
    d = d0.assign(part=d0.pair.map(part))
    test_seqs = set(d.sha1[d.part == "test"])
    tr = d[(d.part == "train") & ~d.sha1.isin(test_seqs)]
    va = d[(d.part == "val") & ~d.sha1.isin(test_seqs)]
    te = d[d.part == "test"]
    return tr, va, te


# ------------------------------------------------------------------ item 3
def add_run(s, emb, allpairs, sub, d0, seeds):
    preds = []
    for seed in seeds:
        t0 = time.time()
        tr, va, te = split_sets(d0, allpairs, seed)
        n = [(x.drop_duplicates("pair").label == c).sum() for x in (tr, va, te) for c in (0, 1)]
        if min(n) < 2:
            continue
        vocab = {g: j for j, g in enumerate(sorted(set(tr.hgene) | set(tr.tgene)))}
        ytr, wtr, yva, wva = tr.label.to_numpy(), T.weights(tr), va.label.to_numpy(), T.weights(va)
        S = [T.features("shortcuts", x, vocab, None) for x in (tr, va, te)]
        ps_va, ps_te, Cs = fit_lr_val_test(S[0], ytr, wtr, S[1], va, S[2])
        preds.append(T.pair_scores(te, ps_te).assign(subset=sub, seed=seed, model="shortcuts", C=Cs))
        for k in KEYS:
            E = [emb[k][x.index.to_numpy()] for x in (tr, va, te)]
            pca = PCA(n_components=32, random_state=0).fit(E[0])
            Z = [np.hstack([S[i], pca.transform(E[i])]) for i in range(3)]
            p, Cc = T.fit_logreg(Z[0], ytr, wtr, Z[1], va, Z[2])
            preds.append(T.pair_scores(te, p).assign(subset=sub, seed=seed, model=f"{k}_pca32+shortcuts", C=Cc))
            pe_va, pe_te, Ce = fit_lr_val_test(E[0], ytr, wtr, E[1], va, E[2])
            st = LogisticRegression(max_iter=2000).fit(np.c_[logit(ps_va), logit(pe_va)], yva, sample_weight=wva)
            p = st.predict_proba(np.c_[logit(ps_te), logit(pe_te)])[:, 1]
            preds.append(T.pair_scores(te, p).assign(subset=sub, seed=seed, model=f"{k}_stack", C=Ce))
        print(f"add {sub} seed {seed}: {time.time() - t0:.0f}s", flush=True)
    return preds


# ------------------------------------------------------------------ item 4
TARGETS = ["label", "long", "rt", "inter", "role_any", "translated"]


def probe_run(s, emb, allpairs, seeds):
    d0 = s.copy()
    d0["long"] = (d0.log_len > d0.log_len.median()).astype(int)
    d0["role_any"] = ((d0.role_h > 0) | (d0.role_t > 0)).astype(int)
    d0["translated"] = d0.A_pipeline.astype(int)
    rows = []
    for seed in seeds:
        t0 = time.time()
        tr, va, te = split_sets(d0, allpairs, seed)
        for tgt in TARGETS:
            # score each sequence on its own: the probe target replaces the label, and each sequence is its own unit
            def unit(x):
                return x.assign(label=x[tgt].astype(int), pair=x.sha1 + "|" + x.pair)
            trt, vat, tet = unit(tr), unit(va), unit(te)
            if min(trt.label.nunique(), vat.label.nunique(), tet.label.nunique()) < 2:
                continue
            for k in KEYS:
                E = [emb[k][x.index.to_numpy()] for x in (trt, vat, tet)]
                p, Cc = T.fit_logreg(E[0], trt.label.to_numpy(), T.weights(trt), E[1], vat, E[2])
                rows.append(pd.DataFrame({"seed": seed, "target": tgt, "model": k, "pair": tet.pair.to_numpy(),
                                          "hgene": tet.hgene.to_numpy(), "tgene": tet.tgene.to_numpy(),
                                          "label": tet.label.to_numpy(), "p": p}))
        print(f"probe seed {seed}: {time.time() - t0:.0f}s", flush=True)
    return pd.concat(rows)


def boot_rows(pr, genes, models, comps, extra):
    wide = pr.pivot_table(index=["seed", "pair", "hgene", "tgene", "label"], columns="model", values="p").reset_index()
    blocks = [b.dropna(subset=models) for _, b in wide.groupby("seed")]
    blocks = [b for b in blocks if 0 < b.label.sum() < len(b)]
    if not blocks:
        return []
    pt, dr, per = RA.gene_boot(blocks, models, genes)
    return RA.summarise(pt, dr, per, models, comps, {**extra, "n_assignments": len(blocks),
                                                      "median_test_units": float(np.median([len(b) for b in blocks]))})


def main():
    t0 = time.time()
    s, emb = T.load()
    s = s.join(C.provenance(s, s), on=["pair", "sha1"])
    allpairs = s.drop_duplicates("pair")[["pair", "hgene", "tgene"]]
    genes = sorted(set(s.hgene) | set(s.tgene))
    genes_of = dict(zip(s.pair, zip(s.hgene, s.tgene)))
    md = ["# Round 3: what embeddings add and encode\n"]

    # item 3
    preds = []
    pooled_seeds = [1, 2] if QUICK else list(range(1, 51))
    single_seeds = [42, 1] if QUICK else [42, *range(1, 21)]
    for sub in ("AB_all", "AB_nonRT"):
        preds += add_run(s, emb, allpairs, sub, (s if sub == "AB_all" else s[s.rt == 0]).copy(), pooled_seeds)
    for sub in M.SUBSETS:
        preds += add_run(s, emb, allpairs, sub, M.subset(s, sub).copy(), single_seeds)
    pr = pd.concat(preds).reset_index()
    pr.to_csv(OUT / "embed_add_predictions.csv.gz", index=False)
    pr = T.mark_both_unseen(pr, s)
    pr["hgene"] = pr.pair.map(lambda x: genes_of[x][0]); pr["tgene"] = pr.pair.map(lambda x: genes_of[x][1])
    models = ["shortcuts"] + [f"{k}_{v}" for k in KEYS for v in ("pca32+shortcuts", "stack")]
    comps = [(m, "shortcuts") for m in models[1:]]
    rows = []
    for sub, q in pr.groupby("subset"):
        for label, both in [("all test pairs", False), ("both genes unseen", True)]:
            z = q[q.both_unseen.astype(bool)] if both else q
            rows += boot_rows(z, genes, models, comps, {"subset": sub, "test_pairs": label})
    add = pd.DataFrame(rows)
    sh = add[(add.kind == "model") & (add.name == "shortcuts")].set_index(["subset", "test_pairs"]).AUROC
    add["headroom_share"] = [r.AUROC / (1 - sh.get((r.subset, r.test_pairs), np.nan)) if r.kind == "difference" else np.nan
                             for r in add.itertuples()]
    add.to_csv(OUT / "embed_add.csv", index=False)
    md += ["## Item 3: embeddings added to the shortcut features (PCA-32 and stacking)", "",
           add[["subset", "test_pairs", "kind", "name", "AUROC", "ci_lo", "ci_hi", "wins", "headroom_share"]]
           .round(3).to_markdown(index=False), ""]

    # item 4
    pb = probe_run(s, emb, allpairs, [42, 1] if QUICK else T.SEEDS)
    pb.to_csv(OUT / "embed_probes_predictions.csv.gz", index=False)
    rows = []
    for tgt, q in pb.groupby("target"):
        rows += boot_rows(q, genes, KEYS, [("fuson", "esm2")], {"target": tgt})
    probes = pd.DataFrame(rows)
    shares = {t: float(s.assign(long=(s.log_len > s.log_len.median()).astype(int),
                                role_any=((s.role_h > 0) | (s.role_t > 0)).astype(int),
                                translated=s.A_pipeline.astype(int))[t].mean()) for t in TARGETS}
    probes["positive_share_all_sequences"] = probes.target.map(shares)
    probes.to_csv(OUT / "embed_probes.csv", index=False)
    md += ["## Item 4: probes on held-out genes (per test sequence)", "",
           probes[["target", "kind", "name", "AUROC", "ci_lo", "ci_hi", "wins", "positive_share_all_sequences"]]
           .round(3).to_markdown(index=False), ""]
    (OUT / "round3_embed.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
