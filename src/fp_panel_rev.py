"""Revision: paediatric panel scored by models trained on FusionPath-style labels (as published, HGNC-normalised).

Replaces score_panel.py (single calibration) and the mean-pooling part of pooling_repeats.py:
  * 47 panel fusions (parser fixed: comma-separated FusOn-DB names, case-insensitive synonyms);
  * 5 calibration repeats (15% of gene pairs held out, seeds 101-105), the same for every model;
  * models: gene names, gene roles, logistic regression and MLP on each embedding (ESM-2 650M, FusOn-pLM with 1,022 and
    2,000-residue windows, ESM-2 3B, ProtT5) and the FusionPath-style model (FusOn-pLM + ProtBERT wild-type partners);
  * an isoform is called when its score is strictly above the 90%-specificity threshold on calibration benign pairs
    (so a constant default score equal to the threshold is not a call); a fusion when more than half its isoforms are;
  * discrimination: AUROC of fusion-level mean scores against held-out calibration benign pairs.
Outputs: results/fp_rev/panel/{isoform_scores.csv.gz, fusion_calls.csv, calibration.csv, summary.md}
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

P = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fp_benchmark_rev as B  # noqa: E402
from build_labels_v2 import norm  # noqa: E402
from fusondb_paediatric_coverage import PANEL, names  # noqa: E402
from train_seq_models import train_mlp  # noqa: E402

OUT = P / "results/fp_rev/panel"
TAGS = ["esm2_650m", "fuson_plm", "fuson_plm_native", "esm2_3b", "prot_t5"]


def panel_emb(tag, sha1s):
    """Panel isoforms: new embeddings include them in the main directory; the first two models in <tag>_panel."""
    for d in [P / f"data/embeddings/{tag}", P / f"data/embeddings/{tag}_panel"]:
        if (d / "index.csv").exists():
            ix = pd.read_csv(d / "index.csv")
            lut = dict(zip(ix.seq_sha1, ix.seq_idx))
            if set(sha1s) <= set(lut):
                E = np.load(d / "emb.npy", mmap_mode="r")
                return np.asarray(E[[lut[s] for s in sha1s]], dtype=np.float32)
    raise SystemExit(f"{tag}: panel isoforms not embedded")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = B.load("as_published")
    emb = {t: B.load_emb(t) for t in TAGS}
    W, wlut = B.load_emb("protbert_wt")
    wt = B.wt_lookup()
    d = d[np.logical_and.reduce([d.sha1.isin(emb[t][1]).to_numpy() for t in TAGS])]
    panel_pairs = {f"{norm(hh)}::{norm(tt)}" for h, t, _ in PANEL for hh in names(h) for tt in names(t)}
    n0 = len(d)
    d = d[~d.pair.isin(panel_pairs)]
    print(f"training data: {len(d):,} records ({n0 - len(d)} panel records removed)", flush=True)

    ps = pd.read_csv(P / "derived/panel_isoforms.csv")
    with open(P / "derived/panel_isoforms.fasta") as f:
        lines = f.read().split("\n")
    seq_of = {lines[i][1:]: lines[i + 1] for i in range(0, len(lines) - 1, 2) if lines[i].startswith(">")}
    ps["seq"] = ps.sha1.map(seq_of)
    ps["raw_h"], ps["raw_t"] = ps.hgene.str.upper(), ps.tgene.str.upper()
    ps["hgene"], ps["tgene"] = ps.raw_h.map(norm), ps.raw_t.map(norm)
    ps["length"] = ps.seq.str.len()
    # gene-level annotation flags (oncogene, TSG, ClinGen, CIViC), taken from the training records on either side
    for k in ("onc", "tsg", "clingen", "civic"):
        flag = pd.concat([d.groupby("hgene")[f"h_{k}"].max(), d.groupby("tgene")[f"t_{k}"].max()]).groupby(level=0).max()
        ps[f"h_{k}"] = ps.hgene.map(flag).fillna(0).astype(int)
        ps[f"t_{k}"] = ps.tgene.map(flag).fillna(0).astype(int)
    role_cols = [f"{s}_{k}" for s in "ht" for k in ("onc", "tsg", "clingen", "civic")]
    pemb = {t: panel_emb(t, ps.sha1.tolist()) for t in TAGS}
    Ef, lutf = emb["fuson_plm_native"]
    P_fpl = B.fusionpath_like(ps, pemb["fuson_plm_native"], {s: i for i, s in enumerate(ps.sha1)}, W, wlut, wt)

    pairs = d.pair.to_numpy()
    upairs = pd.unique(pairs)
    iso, cal_rows = [], []
    for rep in range(101, 106):
        rng = np.random.default_rng(rep)
        cal_pairs = set(rng.choice(upairs, size=int(0.15 * len(upairs)), replace=False))
        m = np.isin(pairs, list(cal_pairs))
        tr, cal = d[~m], d[m]
        ytr, ycal = tr.label.to_numpy(), cal.label.to_numpy()
        scores = {}
        enc = OneHotEncoder(handle_unknown="ignore").fit(tr[["hgene", "tgene"]])
        g = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0).fit(enc.transform(tr[["hgene", "tgene"]]), ytr)
        scores["gene_name"] = (g.predict_proba(enc.transform(cal[["hgene", "tgene"]]))[:, 1],
                               g.predict_proba(enc.transform(ps[["hgene", "tgene"]]))[:, 1])
        r_ = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0).fit(tr[role_cols].to_numpy(float), ytr)
        scores["gene_role"] = (r_.predict_proba(cal[role_cols].to_numpy(float))[:, 1], r_.predict_proba(ps[role_cols].to_numpy(float))[:, 1])
        for t in TAGS:
            E, lut = emb[t]
            X = lambda Z: E[[lut[s] for s in Z.sha1]]  # noqa: E731
            sc = StandardScaler().fit(X(tr))
            Xtr, Xc, Xp = sc.transform(X(tr)).astype(np.float32), sc.transform(X(cal)).astype(np.float32), sc.transform(pemb[t]).astype(np.float32)
            lr = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1).fit(Xtr, ytr)
            scores[f"{t}/logreg"] = (lr.predict_proba(Xc)[:, 1], lr.predict_proba(Xp)[:, 1])
            both = np.mean([train_mlp(Xtr, ytr, Xc, ycal, np.vstack([Xc, Xp]), s) for s in (0, 1, 2)], axis=0)
            scores[f"{t}/mlp"] = (both[:len(Xc)], both[len(Xc):])
        Z = [B.fusionpath_like(Q, Ef, lutf, W, wlut, wt) for Q in (tr, cal)]
        sc = StandardScaler().fit(Z[0])
        Ztr, Zc, Zp = sc.transform(Z[0]).astype(np.float32), sc.transform(Z[1]).astype(np.float32), sc.transform(P_fpl).astype(np.float32)
        both = np.mean([train_mlp(Ztr, ytr, Zc, ycal, np.vstack([Zc, Zp]), s) for s in (0, 1, 2)], axis=0)
        scores["fusionpath_like/mlp"] = (both[:len(Zc)], both[len(Zc):])
        tr_genes = set(tr.hgene) | set(tr.tgene)
        for name, (cs, pp) in scores.items():
            neg = cs[ycal == 0]
            thr = np.quantile(neg, 0.90)
            fs = pd.Series(pp).groupby(ps.fusion.to_numpy()).mean()
            cal_rows.append({"repeat": rep, "model": name, "cal_auroc": roc_auc_score(ycal, cs), "threshold": thr,
                             "panel_vs_cal_benign_auroc": roc_auc_score(np.r_[np.ones(len(fs)), np.zeros(len(neg))], np.r_[fs.to_numpy(), neg]),
                             "isoforms_at_threshold": int(np.isclose(pp, thr).sum())})
            iso.append(ps[["fusion", "disease", "sha1", "length"]].assign(
                repeat=rep, model=name, score=pp, called=pp > thr,
                genes_seen=ps.hgene.isin(tr_genes).astype(int) + ps.tgene.isin(tr_genes).astype(int)))
        print(f"repeat {rep}: {len(tr):,} train / {len(cal):,} calibration records, {time.time() - t0:.0f}s", flush=True)

    iso = pd.concat(iso)
    iso.to_csv(OUT / "isoform_scores.csv.gz", index=False)
    cb = pd.DataFrame(cal_rows)
    cb.to_csv(OUT / "calibration.csv", index=False)
    fus = iso.groupby(["model", "repeat", "fusion"]).agg(share=("called", "mean"), median_len=("length", "median"),
                                                         genes_seen=("genes_seen", "max")).reset_index()
    fus["called"] = fus.share > 0.5
    fus.to_csv(OUT / "fusion_calls.csv", index=False)
    per = fus.groupby(["model", "repeat"]).agg(called=("called", "sum"),
                                               long_called=("called", lambda x: int((x & (fus.loc[x.index, "median_len"] > 1500)).sum())),
                                               called_with_a_seen_gene=("called", lambda x: int((x & (fus.loc[x.index, "genes_seen"] > 0)).sum())))
    tab = per.groupby("model").agg(called_mean=("called", "mean"), called_min=("called", "min"), called_max=("called", "max"),
                                   long_called_mean=("long_called", "mean"), called_with_a_seen_gene=("called_with_a_seen_gene", "mean"))
    tab = tab.join(cb.groupby("model")[["cal_auroc", "panel_vs_cal_benign_auroc", "isoforms_at_threshold"]].mean())
    n_f, n_long = fus.fusion.nunique(), fus.drop_duplicates("fusion").median_len.gt(1500).sum()
    (OUT / "summary.md").write_text(
        f"# Paediatric panel, FusionPath-style labels (revision)\n\n{n_f} fusions ({n_long} longer than 1,500 aa), "
        f"{iso.sha1.nunique()} isoforms; 5 calibration repeats; strict threshold.\n\n" + tab.round(3).to_markdown() + "\n")
    print((OUT / "summary.md").read_text())
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
