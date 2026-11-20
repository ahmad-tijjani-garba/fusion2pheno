"""Step 2 of docs/label_set_plan.md: fusion-pair labels that do not use gene roles, plus an audit.

Positive (evidence about the fusion itself):
  tier1 : CIViC fusion variant with accepted, supporting evidence at level A-C; or CTAT ChimerKB tag
  tier2 : Mitelman, >= 2 independent cases (distinct RefNo/InvNo); sensitivity at >= 3
Negative: recurrent in normal tissue (CTAT GTEx_recurrent_StarF2019 / BodyMap / Greger_Normal /
  Babiceanu_Normal, or Singh 2020 Table S3) AND no cancer evidence of any strength in either orientation
  (CIViC any level, ChimerKB, ChimerPub, chimerdb_pubmed/omim, Cosmic, HaasMedCancer, Mitelman >= 1 case).
Everything else is unlabelled. Positive evidence wins over normal-tissue recurrence (those are counted).
Gene-level tags and tumour RNA-seq observation tags (TCGA/CCLE/ChimerSeq/...) are not used as evidence.
Gene roles are read only for the audit, never for labels.

Outputs: data/labels_v2/pair_evidence.csv, data/labels_v2/row_labels.csv.gz, results/label_audit.md
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import gzip
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

P = F2P_ROOT
E = P / "data/labels_external"
OUT = P / "data/labels_v2"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusondb_paediatric_coverage import PANEL, names  # noqa: E402

NORMAL_CTAT = ["GTEx_recurrent_StarF2019", "BodyMap", "Greger_Normal", "Babiceanu_Normal"]
CANCER_ANY_CTAT = ["ChimerKB", "ChimerPub", "chimerdb_pubmed", "chimerdb_omim", "Cosmic", "HaasMedCancer", "Mitelman"]
READTHROUGH_MAX_BP = 200_000


def sym(s):
    return str(s).strip().upper()


def key(h, t):
    return f"{sym(h)}::{sym(t)}"


def rev(k):
    h, t = k.split("::", 1)
    return f"{t}::{h}"


# ---------------------------------------------------------------- sources
def civic():
    v = pd.read_csv(E / "civic_nightly/VariantSummaries/nightly-VariantSummaries.tsv", sep="\t", dtype=str)
    mp = pd.read_csv(E / "civic_nightly/MolecularProfileSummaries/nightly-MolecularProfileSummaries.tsv", sep="\t", dtype=str)
    ev = pd.read_csv(E / "civic_nightly/AcceptedClinicalEvidenceSummaries/nightly-AcceptedClinicalEvidenceSummaries.tsv",
                     sep="\t", dtype=str)
    fu = v[(v.feature_type == "Fusion") & (v["5_prime_partner_status"] == "known") & (v["3_prime_partner_status"] == "known")]
    var2pair = {r.variant_id: key(r["5_prime_partner"], r["3_prime_partner"]) for _, r in fu.iterrows()}
    mp2pairs = defaultdict(set)
    for _, r in mp.iterrows():
        for vid in str(r.variant_ids).replace(" ", "").split(","):
            if vid in var2pair:
                mp2pairs[r.molecular_profile_id].add(var2pair[vid])
    best = {}  # pair -> best supporting level
    anyev = set(var2pair.values())
    for _, r in ev[ev.evidence_direction == "Supports"].iterrows():
        for k in mp2pairs.get(r.molecular_profile_id, ()):
            best[k] = min(best.get(k, "Z"), r.evidence_level)
    print(f"CIViC: {len(set(var2pair.values()))} fusion pairs with known partners; "
          f"{sum(l <= 'C' for l in best.values())} with supporting A-C evidence", flush=True)
    return best, anyev


def mitelman():
    m = pd.read_csv(E / "mitelman/extracted/MBCA.TXT.DATA", sep="\t", dtype=str, encoding="latin-1")
    cases, refs = defaultdict(set), defaultdict(set)
    for ref, inv, g in zip(m.RefNo, m.InvNo, m.GeneShort.fillna("")):
        for item in g.split(","):
            parts = [sym(x) for x in item.split("::")]
            if len(parts) < 2 or any(not x or x.endswith("+") for x in parts):
                continue
            for h, t in zip(parts[:-1], parts[1:]):  # three-way fusions -> adjacent pairs
                cases[f"{h}::{t}"].add((ref, inv))
                refs[f"{h}::{t}"].add(ref)
    print(f"Mitelman: {len(cases):,} pairs; {sum(len(c) >= 2 for c in cases.values()):,} with >= 2 cases", flush=True)
    return {k: len(c) for k, c in cases.items()}, {k: len(r) for k, r in refs.items()}


def ctat():
    tags = {}
    want = set(NORMAL_CTAT) | set(CANCER_ANY_CTAT)
    with gzip.open(E / "ctat_fusionlib/fusion_lib.Mar2021.dat.gz", "rt") as f:
        for line in f:
            k, _, rest = line.rstrip("\n").partition("\t")
            if "--" not in k:
                continue
            a = rest.partition("\t")[0]
            if not any(w in a for w in want):
                continue
            h, t = re.sub(r"^[^A-Za-z0-9]+", "", k).split("--", 1)
            tags[key(h, t)] = set(json.loads(a)) & want
    print(f"CTAT: {len(tags):,} pairs with a normal-tissue or cancer-evidence tag", flush=True)
    return tags


def singh():
    s3 = pd.read_excel(E / "singh2020/extracted/supp/Table S.xlsx", sheet_name="S3", header=None, dtype=str)
    hdr = next(i for i in range(40) if s3.iloc[i].astype(str).str.fullmatch("GeneName1").any())
    s3.columns = s3.iloc[hdr].astype(str)
    s3 = s3.iloc[hdr + 1:].dropna(subset=["GeneName1", "GeneName2"])
    ft = s3["fusiontype"] if "fusiontype" in s3.columns else pd.Series("", index=s3.index)
    out = {}
    for h, t, f in zip(s3.GeneName1, s3.GeneName2, ft):
        out.setdefault(key(h, t), set()).add(str(f))
    print(f"Singh 2020 S3: {len(s3):,} chimeras, {len(out):,} gene pairs; columns {list(s3.columns)[:14]}", flush=True)
    return out


# ---------------------------------------------------------------- labels
def pair_table():
    civ_best, civ_any = civic()
    mit_cases, mit_refs = mitelman()
    ct = ctat()
    sg = singh()
    pairs = set(civ_any) | set(mit_cases) | set(ct) | set(sg)
    rows = []
    for k in pairs:
        t = ct.get(k, set())
        rows.append({
            "pair": k,
            "civic_best_level": civ_best.get(k, ""),
            "civic_any": k in civ_any,
            "chimerkb": "ChimerKB" in t,
            "mitelman_cases": mit_cases.get(k, 0),
            "mitelman_refs": mit_refs.get(k, 0),
            "ctat_normal": ";".join(sorted(t & set(NORMAL_CTAT))),
            "ctat_cancer_any": ";".join(sorted(t & set(CANCER_ANY_CTAT))),
            "singh_s3": k in sg,
            "singh_type": ";".join(sorted(sg.get(k, ()))),
        })
    d = pd.DataFrame(rows).set_index("pair")
    d["tier1"] = d.civic_best_level.isin(["A", "B", "C"]) | d.chimerkb
    d["tier2"] = d.mitelman_cases >= 2
    d["positive"] = d.tier1 | d.tier2
    d["positive_ge3"] = d.tier1 | (d.mitelman_cases >= 3)
    d["normal"] = (d.ctat_normal != "") | d.singh_s3
    d["cancer_any"] = d.civic_any | (d.ctat_cancer_any != "") | (d.mitelman_cases >= 1)
    ca = set(d.index[d.cancer_any])
    d["cancer_any_either"] = [k in ca or rev(k) in ca for k in d.index]
    d["negative"] = d.normal & ~d.cancer_any_either
    d["label_v2"] = np.where(d.positive, 1, np.where(d.negative, 0, -1))
    return d


def readthrough(rows):
    """Pair-level read-through call from hg38 breakpoints: same chromosome and strand, head upstream, < 200 kb."""
    r = rows.dropna(subset=["hgene_chr", "tgene_chr", "hgene_bp", "tgene_bp"]).copy()
    r["dist"] = np.where(r.hgene_chr == r.tgene_chr, (r.tgene_bp - r.hgene_bp).abs(), np.inf)
    up = np.where(r.hgene_strand == "+", r.hgene_bp < r.tgene_bp, r.hgene_bp > r.tgene_bp)
    r["rt"] = (r.hgene_chr == r.tgene_chr) & (r.hgene_strand == r.tgene_strand) & up & (r.dist < READTHROUGH_MAX_BP)
    g = r.groupby("pair")
    return pd.DataFrame({"min_dist_bp": g.dist.min(), "readthrough_bp": g.rt.mean() >= 0.5,
                         "same_chr": g.apply(lambda x: (x.hgene_chr == x.tgene_chr).any(), include_groups=False)})


def gene_disjoint_test(pairs_df, seed, p=(0.70, 0.15, 0.15)):
    """Same rule as train_seq_models.gene_disjoint_ids, at pair level: pairs whose two genes both fall in fold 2."""
    rng = np.random.default_rng(seed)
    genes = np.array(sorted(set(pairs_df.hgene) | set(pairs_df.tgene)))
    a = dict(zip(genes, rng.choice(3, size=len(genes), p=p)))
    h, t = pairs_df.hgene.map(a), pairs_df.tgene.map(a)
    return pairs_df[(h == t) & (h == 2)], pairs_df[(h == t) & (h == 0)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pe = pair_table()
    cols = ["hgene", "tgene", "aa_seq", "cancer_label", "label_reason", "primary_source", "hgene_role", "tgene_role",
            "hgene_chr", "hgene_bp", "hgene_strand", "tgene_chr", "tgene_bp", "tgene_strand"]
    d = pd.read_csv(P / "data/fusionpath_rebuild/01_source_data/merged/merged_fusions.filtered.hg38.formatted.labeled.csv",
                    usecols=cols, low_memory=False)
    d["row_id"] = np.arange(len(d))
    for c in ["hgene", "tgene"]:
        d[c] = d[c].fillna("?").astype(str).str.upper()
    d["pair"] = d.hgene + "::" + d.tgene
    d["sha1"] = [hashlib.sha1(str(s).replace("*", "").upper().encode()).hexdigest() if pd.notna(s) else None
                 for s in d.aa_seq]
    emb = set(pd.read_csv(P / "data/embeddings/esm2_650m/index.csv").seq_sha1)
    d["embedded"] = d.sha1.isin(emb)
    rt = readthrough(d)
    pe = pe.join(rt, how="left")
    d = d.join(pe[["label_v2", "tier1", "tier2", "positive_ge3", "normal", "mitelman_cases", "civic_best_level",
                   "chimerkb", "singh_s3", "ctat_normal"]], on="pair")
    d["label_v2"] = d.label_v2.fillna(-1).astype(int)
    d = d.join(rt, on="pair")
    panel_pairs = {f"{hh}::{tt}" for h, t, _ in PANEL for hh in names(h) for tt in names(t)}
    d["panel"] = d.pair.isin(panel_pairs)
    d["role_any"] = d.hgene_role.notna() | d.tgene_role.notna()
    pe.to_csv(OUT / "pair_evidence.csv")
    d.drop(columns=["aa_seq"]).to_csv(OUT / "row_labels.csv.gz", index=False)

    # ------------------------------------------------------------ audit
    L = d[(d.label_v2 >= 0) & d.embedded & ~d.panel]
    Lp = L.drop_duplicates("pair")
    T = []
    w = T.append
    w("# Label set v2 audit (step 2 of docs/label_set_plan.md)\n")
    w("Labels are per fusion pair (5'::3', upper-case symbols, exact match; no alias mapping yet). "
      "Gene roles are not used to assign labels; they appear below only as an audit.\n")
    w("## 1. Pair-level evidence (all sources, before matching to sequences)\n")
    s = pd.Series({"pairs in any source": len(pe), "tier 1 (CIViC A-C or ChimerKB)": int(pe.tier1.sum()),
                   "  CIViC A-C": int(pe.civic_best_level.isin(list("ABC")).sum()), "  ChimerKB": int(pe.chimerkb.sum()),
                   "tier 2 (Mitelman >= 2 cases)": int(pe.tier2.sum()), "positive (tier 1 or 2)": int(pe.positive.sum()),
                   "positive, Mitelman >= 3 variant": int(pe.positive_ge3.sum()),
                   "normal-tissue recurrent": int(pe.normal.sum()),
                   "  of which also positive (positive wins)": int((pe.normal & pe.positive).sum()),
                   "  of which some cancer evidence but not positive (unlabelled)": int((pe.normal & ~pe.positive & pe.cancer_any_either).sum()),
                   "negative": int(pe.negative.sum())})
    w(s.to_frame("n").to_markdown() + "\n")
    both = pe[pe.normal & pe.positive].sort_values("mitelman_cases", ascending=False)
    w("Normal-tissue recurrent pairs that are also positive (top 25 by Mitelman cases):\n")
    w(both[["civic_best_level", "chimerkb", "mitelman_cases", "ctat_normal", "singh_s3"]].head(25).to_markdown() + "\n")

    w("## 2. Matched to FusionPath sequences\n")
    m = pd.DataFrame({
        "rows": d.groupby("label_v2").size(), "pairs": d.groupby("label_v2").pair.nunique(),
        "unique sequences": d.groupby("label_v2").sha1.nunique(),
        "rows embedded": d[d.embedded].groupby("label_v2").size()}).rename(index={-1: "unlabelled", 0: "negative", 1: "positive"})
    w(m.to_markdown() + "\n")
    w(f"Panel fusions removed from the labelled set: {int((d.panel & (d.label_v2 >= 0)).sum())} rows, "
      f"{d[d.panel & (d.label_v2 >= 0)].pair.nunique()} pairs. Labelled, embedded, non-panel: "
      f"{len(L):,} rows, {L.pair.nunique():,} pairs ({int((Lp.label_v2 == 1).sum())} positive, "
      f"{int((Lp.label_v2 == 0).sum())} negative).\n")
    w("Positive pairs by tier (labelled, embedded, non-panel):\n")
    w(pd.Series({"tier 1 only": int((Lp.tier1 & ~Lp.tier2).sum()), "tier 2 only": int((~Lp.tier1 & Lp.tier2).sum()),
                 "both": int((Lp.tier1 & Lp.tier2).sum()),
                 "positive with Mitelman >= 3 variant": int(((Lp.label_v2 == 1) & Lp.positive_ge3).sum())}).to_frame("pairs").to_markdown() + "\n")

    w("## 3. Old labels vs new labels (rows, all embedded rows)\n")
    w(pd.crosstab(d[d.embedded].cancer_label, d[d.embedded].label_v2.map({-1: "unlabelled", 0: "negative", 1: "positive"}),
                  margins=True).to_markdown() + "\n")
    w("Pair level (labelled pairs only):\n")
    pl = L.groupby("pair").agg(old=("cancer_label", lambda x: "/".join(sorted(set(x)))), new=("label_v2", "first"))
    w(pd.crosstab(pl.old, pl.new.map({0: "negative", 1: "positive"}), margins=True).to_markdown() + "\n")

    w("## 4. How much do the new labels still follow gene roles?\n")
    w("Share of labelled pairs where either partner has an oncogene/TSG role (the FusionPath rule):\n")
    rr = Lp.groupby("label_v2").role_any.agg(["sum", "count", "mean"]).rename(index={0: "negative", 1: "positive"})
    w(rr.round(3).to_markdown() + "\n")
    w("A gene-role 'classifier' (predict positive iff a partner has a role) on these labels: "
      f"sensitivity {Lp[Lp.label_v2 == 1].role_any.mean():.3f}, specificity {1 - Lp[Lp.label_v2 == 0].role_any.mean():.3f}. "
      "On FusionPath's own labels both are 1.0 by construction.\n")

    w("## 5. Hub genes among positive pairs\n")
    pos = Lp[Lp.label_v2 == 1]
    g = pd.concat([pos.hgene, pos.tgene]).value_counts()
    w(f"{len(pos)} positive pairs, {g.size} distinct genes. Top 15 genes (pairs involving the gene):\n")
    top = g.head(15)
    w(pd.DataFrame({"pairs": top, "share": (top / len(pos)).round(3)}).to_markdown() + "\n")
    w(f"Share of positive pairs touching one of the top 5 genes: "
      f"{pos.pair[(pos.hgene.isin(g.index[:5])) | (pos.tgene.isin(g.index[:5]))].nunique() / len(pos):.3f}\n")

    w("## 6. Read-through share\n")
    rt_tab = Lp.groupby("label_v2").agg(pairs=("pair", "size"), readthrough=("readthrough_bp", "sum"),
                                        same_chr=("same_chr", "sum"), singh_s3=("singh_s3", "sum"))
    rt_tab["readthrough_share"] = rt_tab.readthrough / rt_tab.pairs
    w(rt_tab.rename(index={0: "negative", 1: "positive"}).round(3).to_markdown() + "\n")
    w(f"Read-through = same chromosome and strand, head upstream, breakpoints < {READTHROUGH_MAX_BP // 1000} kb "
      "(majority of a pair's rows). Distance distribution (min breakpoint distance per pair, same-chromosome pairs only):\n")
    q = Lp[Lp.same_chr == True].groupby("label_v2").min_dist_bp.describe(percentiles=[.1, .25, .5, .75, .9])  # noqa: E712
    w(q.rename(index={0: "negative", 1: "positive"}).round(0).to_markdown() + "\n")

    w("## 7. Gene-disjoint power check (go/no-go: >= 40 positive pairs in each test fold)\n")
    lp = Lp[["pair", "hgene", "tgene", "label_v2", "readthrough_bp"]]
    rows = []
    for seed in [42, *range(1, 11)]:
        te, tr = gene_disjoint_test(lp, seed)
        nr = te[te.readthrough_bp != True]  # noqa: E712
        rows.append({"seed": seed, "test_pos": int((te.label_v2 == 1).sum()), "test_neg": int((te.label_v2 == 0).sum()),
                     "test_neg_non_readthrough": int((nr.label_v2 == 0).sum()),
                     "train_pos": int((tr.label_v2 == 1).sum()), "train_neg": int((tr.label_v2 == 0).sum())})
    gd = pd.DataFrame(rows)
    w(gd.to_markdown(index=False) + "\n")
    go = gd.test_pos.min() >= 40
    w(f"**Minimum positive pairs in a test fold: {gd.test_pos.min()} -> {'GO' if go else 'NO-GO'} for the 0.70/0.15/0.15 "
      "gene-disjoint design.**\n")
    if not go:
        rows = []
        for k in (5, 10):
            rng = np.random.default_rng(0)
            genes = np.array(sorted(set(lp.hgene) | set(lp.tgene)))
            a = dict(zip(genes, rng.integers(0, k, size=len(genes))))
            h, t = lp.hgene.map(a), lp.tgene.map(a)
            for f in range(k):
                te = lp[(h == f) & (t == f)]
                rows.append({"k": k, "fold": f, "test_pos": int((te.label_v2 == 1).sum()), "test_neg": int((te.label_v2 == 0).sum())})
        w("Alternative: k-fold gene assignment (pairs kept only when both genes fall in the same fold):\n")
        w(pd.DataFrame(rows).groupby("k")[["test_pos", "test_neg"]].agg(["min", "median", "sum"]).to_markdown() + "\n")

    w("## 8. Symbol matching check\n")
    miss = pe[pe.positive].index.difference(d.pair.unique())
    w(f"{len(miss):,} of {int(pe.positive.sum()):,} positive pairs have no FusionPath sequence under the same symbols. "
      "Some of these will be renamed genes (e.g. MLL->KMT2A, MLLT4->AFDN, C11orf95->ZFTA); an HGNC alias map would recover them.\n")
    old = {"MLL", "MLLT4", "C11ORF95", "MKL1", "WHSC1", "MGEA5", "FAM22A", "FAM22B", "MLLT2", "SEPT9", "SEPT6", "CASC5",
           "TCEB1", "C15ORF21"}
    hits = d[(d.hgene.isin(old)) | (d.tgene.isin(old))]
    w(f"FusionPath rows using a few known outdated symbols: {len(hits):,} rows, genes {sorted(set(hits.hgene) & old | set(hits.tgene) & old)}.\n")
    (P / "results/label_audit.md").write_text("\n".join(T))
    print("\n".join(T))


if __name__ == "__main__":
    main()
