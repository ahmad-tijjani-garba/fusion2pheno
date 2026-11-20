"""Score paediatric driver fusions with sequence models trained on FusionPath-style labels.

1. Training data: rebuilt FusionPath labels (as published), with every record of a
   panel fusion removed so scores are not memorised. Pair-grouped 85/15 split:
   85% train, 15% calibration (held-out pairs).
2. Models: ESM-2 and FusOn-pLM embeddings x {logreg, MLP (3 seeds)}, plus the
   gene-name-only baseline for contrast.
3. Panel sequences: every isoform of the 54 panel fusions from FusOn-DB and from the
   rebuilt dataset (most of the latter were dropped as "Uncertain"). New sequences
   are embedded here (inference only, same 1,022-residue overlapping windows).
4. For each fusion: mean score over isoforms, percentile among calibration *benign*
   scores, and the call at the threshold giving 90% specificity on calibration.
Outputs: results/panel_scores.csv (per isoform), results/panel_scores_summary.{csv,md}
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from transformers import AutoModel, AutoTokenizer

P = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_sequences import windows  # noqa: E402  same chunking as the main embeddings
from fusondb_paediatric_coverage import PANEL, names  # noqa: E402
from train_seq_models import load_rows, train_mlp, norm_seq  # noqa: E402

MODELS = {"esm2_650m": "facebook/esm2_t33_650M_UR50D", "fuson_plm": "ChatterjeeLab/FusOn-pLM"}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


@torch.no_grad()
def embed(seqs, model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV).eval()
    out = np.zeros((len(seqs), model.config.hidden_size), np.float32)
    for i, s in enumerate(seqs):
        acc = np.zeros((len(s), model.config.hidden_size), np.float32)
        cnt = np.zeros(len(s), np.float32)
        for start, w in windows(s):
            enc = tok([w], return_tensors="pt").to(DEV)
            h = model(**enc).last_hidden_state[0, 1:len(w) + 1].float().cpu().numpy()
            acc[start:start + len(w)] += h
            cnt[start:start + len(w)] += 1
        out[i] = (acc / cnt[:, None]).mean(0)
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return out


def panel_sequences(rebuilt):
    rows = []
    fdb = pd.read_csv(P / "data/raw/fuson_db/FusOn-DB.csv", usecols=["aa_seq", "fusiongenes"])
    # revision fix: FusOn-DB lists some fusions under several comma-separated names, and old symbols are
    # mixed-case (e.g. C11orf95::RELA for ZFTA::RELA); split the lists and compare symbols case-insensitively
    fdb = fdb.assign(fusiongenes=fdb.fusiongenes.str.split(",")).explode("fusiongenes")
    parts = fdb.fusiongenes.str.strip().str.split("::", n=1, expand=True)
    fdb["hgene"], fdb["tgene"] = parts[0].str.upper(), parts[1].str.upper()
    up = lambda g: {x.upper() for x in names(g)}  # noqa: E731
    for h, t, disease in PANEL:
        for src, df in [("rebuilt", rebuilt), ("FusOn-DB", fdb)]:  # rebuilt first keeps its label on dedup
            sub = df[df.hgene.isin(up(h)) & df.tgene.isin(up(t))]
            for _, r in sub.iterrows():
                if pd.notna(r.aa_seq):
                    rows.append({"fusion": f"{h}::{t}", "disease": disease, "source": src,
                                 "hgene": h, "tgene": t, "aa_seq": norm_seq(r.aa_seq),
                                 "fusionpath_label": r.get("cancer_label", "not in rebuilt data")})
    ps = pd.DataFrame(rows)
    ps["sha1"] = [hashlib.sha1(s.encode()).hexdigest() for s in ps.aa_seq]
    return ps.drop_duplicates(["fusion", "sha1"])


def main():
    full = pd.read_csv(P / "data/fusionpath_rebuild/01_source_data/merged/"
                       "merged_fusions.filtered.hg38.formatted.labeled.csv",
                       low_memory=False, usecols=["hgene", "tgene", "aa_seq", "cancer_label"])
    full["hgene"] = full.hgene.fillna("?").astype(str).str.upper()
    full["tgene"] = full.tgene.fillna("?").astype(str).str.upper()
    ps = panel_sequences(full)
    print(f"panel isoforms to score: {len(ps):,} unique (fusion, sequence) across "
          f"{ps.fusion.nunique()} fusions", flush=True)

    d = load_rows("as_published")
    panel_pairs = {(hh, tt) for h, t, _ in PANEL for hh in names(h) for tt in names(t)}
    is_panel = [(h, t) in panel_pairs for h, t in zip(d.hgene, d.tgene)]
    print(f"removing {sum(is_panel)} labelled panel records from training", flush=True)
    d = d[~np.array(is_panel)]

    rng = np.random.default_rng(SEED)
    pairs = (d.hgene + "::" + d.tgene).unique()
    cal_pairs = set(rng.choice(pairs, size=int(0.15 * len(pairs)), replace=False))
    in_cal = (d.hgene + "::" + d.tgene).isin(cal_pairs).to_numpy()
    tr, cal = d[~in_cal], d[in_cal]
    ytr, ycal = tr.label.to_numpy(), cal.label.to_numpy()

    tr_genes = set(tr.hgene) | set(tr.tgene)
    ps["head_seen_in_training"] = ps.hgene.isin(tr_genes)
    ps["tail_seen_in_training"] = ps.tgene.isin(tr_genes)

    # gene-name baseline
    enc = OneHotEncoder(handle_unknown="ignore").fit(tr[["hgene", "tgene"]])
    gn = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED)
    gn.fit(enc.transform(tr[["hgene", "tgene"]]), ytr)
    cal_scores = {"gene_name": gn.predict_proba(enc.transform(cal[["hgene", "tgene"]]))[:, 1]}
    ps["score_gene_name"] = gn.predict_proba(enc.transform(ps[["hgene", "tgene"]]))[:, 1]

    for tag, model_id in MODELS.items():
        idx = pd.read_csv(P / f"data/embeddings/{tag}/index.csv")
        E = np.load(P / f"data/embeddings/{tag}/emb.npy").astype(np.float32)
        lut = dict(zip(idx.seq_sha1, idx.seq_idx))
        Xtr = E[[lut[h] for h in tr.sha1]]
        Xcal = E[[lut[h] for h in cal.sha1]]
        Xp = embed(ps.aa_seq.tolist(), model_id)
        print(f"{tag}: embedded {len(Xp)} panel sequences", flush=True)
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xcal_s, Xp_s = [sc.transform(x).astype(np.float32) for x in (Xtr, Xcal, Xp)]
        lr = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1, random_state=SEED).fit(Xtr_s, ytr)
        cal_scores[f"{tag}_logreg"] = lr.predict_proba(Xcal_s)[:, 1]
        ps[f"score_{tag}_logreg"] = lr.predict_proba(Xp_s)[:, 1]
        # MLP: early stopping on calibration set, then score calibration + panel together
        both = np.vstack([Xcal_s, Xp_s])
        p_both = np.mean([train_mlp(Xtr_s, ytr, Xcal_s, ycal, both, s) for s in (0, 1, 2)], axis=0)
        cal_scores[f"{tag}_mlp"] = p_both[:len(Xcal_s)]
        ps[f"score_{tag}_mlp"] = p_both[len(Xcal_s):]

    summ_rows, cal_info = [], []
    benign = ycal == 0
    for m, cs in cal_scores.items():
        thr = np.quantile(cs[benign], 0.90)  # 90% specificity on held-out pairs
        cal_info.append(f"- {m}: calibration AUROC {roc_auc_score(ycal, cs):.3f}; "
                        f"90%-specificity threshold {thr:.3f}; sensitivity at it {np.mean(cs[~benign] >= thr):.1%}")
        ps[f"pct_benign_{m}"] = [np.mean(cs[benign] < s) * 100 for s in ps[f"score_{m}"]]
        ps[f"call_{m}"] = ps[f"score_{m}"] >= thr
    ps.drop(columns=["aa_seq"]).to_csv(P / "results/panel_scores.csv", index=False)

    for (fusion, disease), g in ps.groupby(["fusion", "disease"], sort=False):
        row = {"fusion": fusion, "disease": disease, "isoforms": len(g),
               "FusionPath label": "; ".join(sorted(set(map(str, g.fusionpath_label)))),
               "genes seen": f"{int(g.head_seen_in_training.iloc[0])}{int(g.tail_seen_in_training.iloc[0])}"}
        for m in cal_scores:
            row[f"{m} pct"] = round(g[f"pct_benign_{m}"].mean())
            row[f"{m} called"] = f"{g[f'call_{m}'].mean():.0%}"
        summ_rows.append(row)
    summ = pd.DataFrame(summ_rows)
    summ.to_csv(P / "results/panel_scores_summary.csv", index=False)

    keep = ["fusion", "disease", "isoforms", "FusionPath label", "genes seen",
            "gene_name pct", "esm2_650m_mlp pct", "esm2_650m_mlp called",
            "fuson_plm_mlp pct", "fuson_plm_mlp called"]
    n = len(summ)
    called = {m: (summ[f"{m} called"].str.rstrip("%").astype(float) >= 50).sum() for m in cal_scores}
    text = ["# Paediatric driver fusions scored by models trained on FusionPath-style labels", "",
            "Panel records removed from training. 'pct' = mean percentile of the fusion's isoform scores among",
            "held-out *benign* calibration scores (100 = scores above every benign). 'called' = share of isoforms",
            "above the 90%-specificity threshold. 'genes seen' = head/tail gene present in training (1/0).", "",
            "## Calibration (held-out fusion pairs)", "", *cal_info, "",
            "## Fusions called pathogenic by majority of isoforms", "",
            *[f"- {m}: {called[m]}/{n}" for m in cal_scores], "",
            "## Per fusion", "", summ[keep].to_markdown(index=False), ""]
    (P / "results/panel_scores_summary.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
