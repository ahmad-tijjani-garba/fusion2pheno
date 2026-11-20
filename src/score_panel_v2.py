"""Score the paediatric driver panel with models trained on the strict v2 labels.

Training data : strict v2 labelled sequences (panel pairs already removed; no panel isoform is identical to a
                training sequence). 5 calibration repeats: 15% of labelled pairs held out (random, pair-grouped),
                85% train. Logreg C and MLP early stopping use the calibration pairs.
Models        : length, gene_name, shortcuts (length + geometry + partner role + gene name), ESM-2 / FusOn-pLM
                (logreg, MLP 3 inits), ESM-2 / FusOn-pLM + shortcuts. Source features are left out: panel isoforms
                all come from databases, so a source feature would call them by provenance.
Panel         : every isoform of the panel fusions from the rebuilt FusionPath data and FusOn-DB (same as
                score_panel.py). Mean-pooled embeddings computed here (inference only) and cached in
                data/embeddings/<tag>_panel/.
Calls         : threshold = 90% specificity on calibration negative pairs; an isoform is called if it scores strictly
                above it; a fusion is called when most of its isoforms are (same rule as score_panel.py).
Outputs       : results/v2_strict/panel/{panel_scores.csv, panel_summary.csv, summary.md}
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

P = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_v2 as T  # noqa: E402
from build_labels_v2 import gene_coords, norm, pair_geometry, pkey  # noqa: E402

OUT = P / "results/v2_strict/panel"
MERGED = P / "data/fusionpath_rebuild/01_source_data/merged/merged_fusions.filtered.hg38.formatted.labeled.csv"


def panel_isoforms():
    from score_panel import panel_sequences
    full = pd.read_csv(MERGED, low_memory=False, usecols=["hgene", "tgene", "aa_seq", "cancer_label"])
    full["hgene"] = full.hgene.fillna("?").astype(str).str.upper()
    full["tgene"] = full.tgene.fillna("?").astype(str).str.upper()
    ps = panel_sequences(full).reset_index(drop=True)
    ps["length"] = ps.aa_seq.str.len()
    ps["hgene"] = ps.hgene.map(norm)
    ps["tgene"] = ps.tgene.map(norm)
    ps["pair"] = [pkey(h, t) for h, t in zip(ps.hgene, ps.tgene)]
    return ps


def panel_embeddings(ps):
    out = {}
    for k, tag in T.TAGS.items():
        u = ps.drop_duplicates("sha1").reset_index(drop=True)
        main_dir = P / f"data/embeddings/{tag}"
        if tag not in ("esm2_650m", "fuson_plm") and (main_dir / "index.csv").exists() and \
                set(u.sha1) <= set(pd.read_csv(main_dir / "index.csv").seq_sha1):
            ix = pd.read_csv(main_dir / "index.csv")
            E = np.load(main_dir / "emb.npy", mmap_mode="r")
            lut = dict(zip(ix.seq_sha1, ix.seq_idx))
            out[k] = np.asarray(E[[lut[s] for s in ps.sha1]], dtype=np.float32)
            continue
        d = P / f"data/embeddings/{tag}_panel"
        if not ((d / "index.csv").exists() and set(u.sha1) <= set(pd.read_csv(d / "index.csv").seq_sha1)):
            from score_panel import MODELS, embed
            t0 = time.time()
            E = embed(u.aa_seq.tolist(), MODELS[tag]).astype(np.float16)
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "emb.npy", E)
            pd.DataFrame({"seq_idx": range(len(u)), "seq_sha1": u.sha1, "length": u.length}).to_csv(d / "index.csv", index=False)
            print(f"{tag}: embedded {len(u)} panel sequences in {time.time() - t0:.0f}s -> {d}", flush=True)
        ix = pd.read_csv(d / "index.csv")
        E = np.load(d / "emb.npy").astype(np.float32)
        lut = dict(zip(ix.seq_sha1, ix.seq_idx))
        out[k] = E[[lut[s] for s in ps.sha1]]
        # consistency with the main embeddings for isoforms embedded there too
        main = {}
        for t in [tag, tag + "_v2new"]:
            if not (P / f"data/embeddings/{t}/index.csv").exists():
                continue
            mi = pd.read_csv(P / f"data/embeddings/{t}/index.csv")
            me = np.load(P / f"data/embeddings/{t}/emb.npy", mmap_mode="r")
            for sh, i in zip(mi.seq_sha1, mi.seq_idx):
                if sh in lut and sh not in main:
                    main[sh] = np.asarray(me[i], dtype=np.float32)
        if main:
            cos = [float(np.dot(main[s], E[lut[s]]) / np.linalg.norm(main[s]) / np.linalg.norm(E[lut[s]])) for s in main]
            print(f"{tag}: {len(main)} panel isoforms also in the main embeddings; cosine min {min(cos):.4f}", flush=True)
    return out


def panel_features(ps):
    gc = gene_coords()
    geo = pair_geometry(set(ps.pair), gc)
    gap = ps.pair.map(lambda k: geo[k][0]).astype(float)
    ps["rt"] = ps.pair.map(lambda k: geo[k][1] is True).astype(float)
    ps["geo_missing"] = gap.isna().astype(float)
    ps["inter"] = np.isinf(gap).astype(float)
    ps["log_gap"] = np.where(np.isfinite(gap), np.log10(gap.clip(lower=0) + 1), 0.0)
    roles = pd.read_csv(MERGED, usecols=["hgene", "tgene", "hgene_role", "tgene_role"], low_memory=False)
    role = {norm(g) for g in pd.concat([roles.hgene[roles.hgene_role.notna()], roles.tgene[roles.tgene_role.notna()]])}
    ps["role_h"] = ps.hgene.isin(role).astype(float)
    ps["role_t"] = ps.tgene.isin(role).astype(float)
    ps["log_len"] = np.log10(ps.length.astype(float))
    return ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--no-mlp", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    s, emb = T.load()
    ps = panel_features(panel_isoforms())
    print(f"panel: {len(ps)} isoforms of {ps.fusion.nunique()} fusions; "
          f"{ps.sha1.isin(set(s.sha1)).sum()} identical to a labelled sequence", flush=True)
    pemb = panel_embeddings(ps)
    lab_genes = set(s.hgene) | set(s.tgene)
    ps["head_in_labels"] = ps.hgene.isin(lab_genes)
    ps["tail_in_labels"] = ps.tgene.isin(lab_genes)

    pairs = np.array(sorted(s.pair.unique()))
    calib, iso_scores = [], []
    for r in range(a.repeats):
        rng = np.random.default_rng(r)
        cal_pairs = set(rng.choice(pairs, size=int(0.15 * len(pairs)), replace=False))
        cal = s[s.pair.isin(cal_pairs)]
        tr = s[~s.pair.isin(cal_pairs) & ~s.sha1.isin(set(cal.sha1))]
        vocab = {g: j for j, g in enumerate(sorted(set(tr.hgene) | set(tr.tgene)))}
        ytr, wtr = tr.label.to_numpy(), T.weights(tr)
        jobs = [(m, None) for m in ["length", "gene_name", "shortcuts"]] + \
               [(m, k) for k in emb for m in ["plm", "plm+shortcuts"]]
        preds = {}
        for m, k in jobs:
            F = [T.features(m, x, vocab, E) for x, E in
                 [(tr, emb[k][tr.index.to_numpy()] if k else None), (cal, emb[k][cal.index.to_numpy()] if k else None),
                  (ps, pemb[k] if k else None)]]
            p, _ = T.fit_logreg(F[0], ytr, wtr, F[1], cal, np.vstack([F[1], F[2]]))
            preds[m if k is None else m.replace("plm", k) + ("_logreg" if m == "plm" else "")] = p
        if not a.no_mlp:
            for k in emb:
                Xc, Xp = emb[k][cal.index.to_numpy()], pemb[k]
                preds[f"{k}_mlp"] = T.fit_mlp(emb[k][tr.index.to_numpy()], ytr, wtr, Xc, cal, np.vstack([Xc, Xp]), r)
        for name, p in preds.items():
            pc, pp = p[:len(cal)], p[len(cal):]
            g = T.pair_scores(cal, pc)
            neg = g.p[g.label == 0].to_numpy()
            thr = np.quantile(neg, 0.90)
            fs = pd.Series(pp).groupby(ps.fusion.to_numpy()).mean()
            calib.append({"repeat": r, "model": name, "cal_auroc": roc_auc_score(g.label, g.p),
                          "panel_vs_cal_negatives_auroc": roc_auc_score(np.r_[np.ones(len(fs)), np.zeros(len(neg))],
                                                                        np.r_[fs.to_numpy(), neg]),
                          "n_ties_at_threshold": int(np.isclose(pp, thr).sum()),
                          "sens_at_90spec": float((g.p[g.label == 1] > thr).mean()), "threshold": thr,
                          "cal_pairs": len(g), "cal_pos": int(g.label.sum())})
            iso_scores.append(ps[["fusion", "sha1"]].assign(repeat=r, model=name, score=pp, called=pp > thr,
                                                            pct_neg=[(neg < x).mean() * 100 for x in pp]))
        print(f"repeat {r}: train {tr.pair.nunique()} pairs, calibration {cal.pair.nunique()} pairs, "
              f"{time.time() - t0:.0f}s", flush=True)

    cb = pd.DataFrame(calib)
    iso = pd.concat(iso_scores)
    iso.to_csv(OUT / "panel_scores.csv", index=False)
    fus = iso.groupby(["model", "repeat", "fusion"]).agg(called_share=("called", "mean"), pct=("pct_neg", "mean")).reset_index()
    fus["called"] = fus.called_share > 0.5
    per_rep = fus.groupby(["model", "repeat"]).called.sum().unstack()
    n_fus = ps.fusion.nunique()

    meta = ps.groupby("fusion", sort=False).agg(disease=("disease", "first"), isoforms=("sha1", "size"),
                                                median_aa=("length", "median"), head=("head_in_labels", "first"),
                                                tail=("tail_in_labels", "first")).reset_index()
    long = set(meta.fusion[meta.median_aa > 1500])
    old = pd.read_csv(F2P_ROOT / "results/panel_scores_summary.csv")
    old_called = {m: set(old.fusion[old[f"{m} called"].str.rstrip("%").astype(float) >= 50])
                  for m in ["esm2_650m_mlp", "fuson_plm_mlp", "gene_name"]}

    order = ["length", "gene_name", "shortcuts", "esm2_logreg", "esm2_mlp", "fuson_logreg", "fuson_mlp",
             "esm2+shortcuts", "fuson+shortcuts"]
    order = [m for m in order if m in per_rep.index]
    rows = []
    for m in order:
        c = cb[cb.model == m]
        f = fus[fus.model == m]
        rows.append({"model": m, "calibration AUROC": c.cal_auroc.mean(), "sensitivity at 90% spec": c.sens_at_90spec.mean(),
                     f"fusions called (of {n_fus}), mean": per_rep.loc[m].mean(),
                     "range over repeats": f"{per_rep.loc[m].min()}-{per_rep.loc[m].max()}",
                     f"long fusions called (of {len(long)})": f[f.fusion.isin(long)].groupby("repeat").called.sum().mean()})
    tab = pd.DataFrame(rows).set_index("model").round(3)

    share = fus.groupby(["fusion", "model"]).called.mean().unstack()
    pct = fus.groupby(["fusion", "model"]).pct.mean().unstack()
    per = meta.set_index("fusion")
    per["genes in labels"] = per["head"].astype(int).astype(str) + per["tail"].astype(int).astype(str)
    for m in ["shortcuts", "esm2_mlp", "fuson_mlp", "esm2+shortcuts"]:
        if m in share:
            per[f"{m} called"] = (share[m] * 100).round().astype(int).astype(str) + "%"
    per["old ESM-2 MLP (FusionPath labels)"] = ["yes" if f in old_called["esm2_650m_mlp"] else "no" for f in per.index]
    per["old FusOn-pLM MLP"] = ["yes" if f in old_called["fuson_plm_mlp"] else "no" for f in per.index]
    per = per.drop(columns=["head", "tail"])
    per.to_csv(OUT / "panel_summary.csv")

    def agree(m_new, m_old):
        new_called = set(share.index[share[m_new] > 0.5])
        o = old_called[m_old]
        return f"{len(new_called & o)} both, {len(o - new_called)} old only, {len(new_called - o)} new only"

    L = ["# Paediatric driver panel scored by models trained on the strict v2 labels\n",
         f"{len(ps)} isoforms of {n_fus} fusions; none identical to a labelled training sequence. "
         f"{args_line(a)} Threshold = 90% specificity on calibration negative pairs; a fusion is called when most of its "
         "isoforms score strictly above it. Long = median isoform > 1,500 aa. Source features left out (panel isoforms all "
         "come from databases).\n",
         tab.to_markdown() + "\n",
         "Old = score_panel.py, trained on FusionPath labels (as published), same calling rule.\n",
         f"- old ESM-2 MLP called {len(old_called['esm2_650m_mlp'])}/{n_fus}, old FusOn-pLM MLP "
         f"{len(old_called['fuson_plm_mlp'])}/{n_fus}, old gene-name {len(old_called['gene_name'])}/{n_fus}",
         f"- ESM-2 MLP, new (called in > half of repeats) vs old: {agree('esm2_mlp', 'esm2_650m_mlp')}" if "esm2_mlp" in share else "",
         f"- FusOn-pLM MLP, new vs old: {agree('fuson_mlp', 'fuson_plm_mlp')}\n" if "fuson_mlp" in share else "",
         "## Per fusion (share of the repeats in which the fusion is called)\n",
         per.sort_values("median_aa").to_markdown() + "\n",
         "## Calibration per repeat\n", cb.round(3).to_markdown(index=False) + "\n"]
    md = "\n".join(L)
    (OUT / "summary.md").write_text(md)
    print(md)
    print(f"total {time.time() - t0:.0f}s")


def args_line(a):
    return f"{a.repeats} calibration repeats (15% of labelled pairs held out each time)."


if __name__ == "__main__":
    main()
