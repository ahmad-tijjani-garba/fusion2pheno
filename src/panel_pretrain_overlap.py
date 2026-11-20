"""Were the paediatric panel fusions in FusOn-pLM's pretraining data, and does that explain its edge over ESM-2?

Uses FusOn-pLM's own split files (bundled in the HF snapshot): fuson_plm/data/splits/{train,val,test}_df.csv
(MMseqs2 30% identity clusters; sequences > 2,000 aa were excluded from all splits).
Matches panel isoforms by exact normalised-sequence SHA1, and fusions by gene pair (with synonyms).
Relates overlap to (a) isoform-level percentile scores from score_panel.py (whole-protein mean, MLP) and
(b) fusion-level call shares over the 5 calibration repeats (pooling_repeats_B_fusions.csv, mean pooling).
Outputs: results/panel_pretrain_overlap_isoforms.csv, results/panel_pretrain_overlap_fusions.csv,
         results/panel_pretrain_overlap.md
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

P = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusondb_paediatric_coverage import names  # noqa: E402
from train_seq_models import norm_seq  # noqa: E402

SNAP = next((P / "models/hf_cache/hub/models--ChatterjeeLab--FusOn-pLM/snapshots").iterdir())
S = SNAP / "fuson_plm/data"


def sha1(s):
    return hashlib.sha1(norm_seq(s).encode()).hexdigest()


def gene_pairs(fg):
    out = []
    for p in re.split(r"[,;|]", str(fg)):
        if "::" in p:
            a, b = p.strip().upper().split("::", 1)
            out.append((a.strip(), b.strip()))
    return out


def mw(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    return mannwhitneyu(a, b).pvalue if len(a) >= 3 and len(b) >= 3 else np.nan


def main():
    split_of = {}
    for sp in ("train", "val", "test"):
        for s in pd.read_csv(S / f"splits/{sp}_df.csv", usecols=["sequence"]).sequence:
            split_of[sha1(s)] = sp
    db = pd.read_csv(S / "fuson_db.csv")
    db["sha1"] = db.aa_seq.map(sha1)
    db["split"] = db.sha1.map(split_of).fillna("excluded")
    print("FusOn-DB rows by FusOn-pLM split:", db.split.value_counts().to_dict(), flush=True)
    print("example fusiongenes:", db.fusiongenes.head(3).tolist(), flush=True)
    pair_split = {}
    for fg, sp in zip(db.fusiongenes, db.split):
        for pr in gene_pairs(fg):
            pair_split.setdefault(pr, set()).add(sp)

    ps = pd.read_csv(P / "results/panel_scores.csv")
    idx = pd.read_csv(P / "data/embeddings_pooled/fuson_plm/index.csv")
    ps["length"] = ps.sha1.map(dict(zip(idx.seq_sha1, idx.length)))
    ps["in_fusondb"] = ps.sha1.isin(set(db.sha1))
    ps["iso_split"] = ps.sha1.map(split_of).fillna("absent")

    def pair_splits(fusion, hg, tg):
        h0, t0 = fusion.split("::", 1)
        hs, ts = names(h0) | {hg}, names(t0) | {tg}
        found = set()
        for a in hs:
            for b in ts:
                found |= pair_split.get((str(a).upper(), str(b).upper()), set())
        return found

    ps["pair_splits"] = [",".join(sorted(pair_splits(f, h, t))) for f, h, t in zip(ps.fusion, ps.hgene, ps.tgene)]
    ps["pair_in_train"] = ps.pair_splits.str.contains("train")
    ps["diff_pct"] = ps.pct_benign_fuson_plm_mlp - ps.pct_benign_esm2_650m_mlp
    ps.drop(columns=[c for c in ps.columns if c.startswith(("score_", "call_gene", "pct_benign_gene"))]).to_csv(
        P / "results/panel_pretrain_overlap_isoforms.csv", index=False)

    # fusion level, with call shares over the 5 calibration repeats (mean pooling)
    bf = pd.read_csv(P / "results/pooling_repeats_B_fusions.csv")
    cs = bf[bf.pooling == "mean"].pivot_table(index="fusion", columns="embedding", values="called", aggfunc="mean")
    fu = ps.groupby("fusion").agg(
        disease=("disease", "first"), isoforms=("sha1", "size"), median_len=("length", "median"),
        iso_train=("iso_split", lambda s: int((s == "train").sum())),
        iso_valtest=("iso_split", lambda s: int(s.isin(["val", "test"]).sum())),
        iso_in_fusondb=("in_fusondb", "sum"), pair_in_train=("pair_in_train", "any"),
        pair_splits=("pair_splits", "first"),
        pct_fuson=("pct_benign_fuson_plm_mlp", "mean"), pct_esm2=("pct_benign_esm2_650m_mlp", "mean"))
    fu["iso_train_share"] = fu.iso_train / fu.isoforms
    fu = fu.join(cs.rename(columns={"esm2_650m": "called5_esm2", "fuson_plm": "called5_fuson"}))
    fu["called5_diff"] = fu.called5_fuson - fu.called5_esm2
    fu["status"] = np.select([fu.iso_train > 0, fu.pair_in_train, fu.pair_splits != ""],
                             ["isoform in train", "pair in train (other isoforms)", "pair only in val/test/excluded"],
                             "pair not in FusOn-DB")
    fu.sort_values(["status", "fusion"]).to_csv(P / "results/panel_pretrain_overlap_fusions.csv")

    # isoform-level comparison
    ps["grp"] = np.where(ps.iso_split == "train", "train",
                         np.where(ps.iso_split.isin(["val", "test"]), "val/test",
                                  np.where(ps.length > 2000, "absent, >2000 aa", "absent, <=2000 aa")))
    iso_tab = ps.groupby("grp").agg(isoforms=("sha1", "size"), fusions=("fusion", "nunique"),
                                    median_len=("length", "median"),
                                    pct_fuson=("pct_benign_fuson_plm_mlp", "mean"),
                                    pct_esm2=("pct_benign_esm2_650m_mlp", "mean"),
                                    diff=("diff_pct", "mean"),
                                    call_fuson=("call_fuson_plm_mlp", "mean"),
                                    call_esm2=("call_esm2_650m_mlp", "mean")).round(3)
    tr = ps[ps.grp == "train"].diff_pct
    p_iso = {g: mw(tr, ps[ps.grp == g].diff_pct) for g in ps.grp.unique() if g != "train"}
    # length-matched: <=2000 aa only
    short = ps[ps.length <= 2000]
    p_short = mw(short[short.iso_split == "train"].diff_pct, short[short.iso_split != "train"].diff_pct)

    # within-fusion: fusions with both train and non-train isoforms
    within = []
    for f, g in ps.groupby("fusion"):
        a, b = g[g.iso_split == "train"].diff_pct, g[g.iso_split != "train"].diff_pct
        if len(a) and len(b):
            within.append({"fusion": f, "n_train": len(a), "n_other": len(b),
                           "diff_train": a.mean(), "diff_other": b.mean()})
    within = pd.DataFrame(within)
    p_within = (wilcoxon(within.diff_train - within.diff_other).pvalue
                if len(within) >= 5 else np.nan)

    fus_tab = fu.groupby("status").agg(fusions=("isoforms", "size"), median_len=("median_len", "median"),
                                       called5_fuson=("called5_fuson", "mean"),
                                       called5_esm2=("called5_esm2", "mean"),
                                       called5_diff=("called5_diff", "mean")).round(3)
    has_train = fu.status == "isoform in train"
    p_fus = mw(fu[has_train].called5_diff, fu[~has_train].called5_diff)

    text = ["# Panel overlap with FusOn-pLM pretraining data", "",
            f"FusOn-pLM splits: {len(split_of):,} sequences. FusOn-DB rows by split: "
            f"{db.split.value_counts().to_dict()} ('excluded' = >2,000 aa or removed during their cleaning).", "",
            f"Panel isoforms: {len(ps)}; exact match in FusOn-DB: {int(ps.in_fusondb.sum())}; "
            f"by FusOn-pLM split: {ps.iso_split.value_counts().to_dict()}.", "",
            "## Isoform level (score_panel.py, whole-protein mean, MLP; pct = percentile among benign calibration)", "",
            "diff = FusOn-pLM pct - ESM-2 pct. Mann-Whitney vs 'train': "
            + ", ".join(f"{k}: p={v:.3g}" for k, v in p_iso.items())
            + f". Length-matched (<=2000 aa only), train vs not: p={p_short:.3g}.", "",
            iso_tab.to_markdown(), "",
            f"Within-fusion (fusions with both train and non-train isoforms, n={len(within)}): "
            f"mean diff train {within.diff_train.mean() if len(within) else np.nan:.2f} vs other "
            f"{within.diff_other.mean() if len(within) else np.nan:.2f}, Wilcoxon p={p_within:.3g}.", "",
            (within.round(2).to_markdown(index=False) if len(within) else "(none)"), "",
            "## Fusion level (call share over 5 calibration repeats, mean pooling)", "",
            f"called5_diff = FusOn-pLM - ESM-2. Mann-Whitney 'isoform in train' vs rest: p={p_fus:.3g}.", "",
            fus_tab.to_markdown(), "",
            fu[["disease", "isoforms", "median_len", "iso_train", "iso_valtest", "iso_in_fusondb", "status",
                "called5_esm2", "called5_fuson", "called5_diff"]].sort_values(["status", "called5_diff"])
            .round(2).to_markdown(), ""]
    (P / "results/panel_pretrain_overlap.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
