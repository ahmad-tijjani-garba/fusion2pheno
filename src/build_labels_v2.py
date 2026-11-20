"""Label set v2 (strict): pair labels with HGNC-normalised symbols, strict Mitelman tier, and sequences from
(A) one translation pipeline (GENCODE v50, canonical exon-edge junctions, in frame) for FusionPath and Singh
breakpoints, and (B) existing database sequences (FusionPath, FusOn-DB, FusionGDB2) as a larger secondary set.

Positive : CIViC accepted supporting evidence A-C, or ChimerKB, or >= 2 Mitelman cases from focused references
           (references with <= 100 fusion cases; the TCGA-scale re-analyses are excluded).
Negative : recurrent in normal tissue (CTAT normal tags or Singh 2020 S3) and no cancer evidence of any strength,
           in either orientation, counting Mitelman cases from all references.
Read-through: same chromosome and strand, 5' gene upstream in transcription direction, gene gap < 200 kb (GENCODE genes).
Outputs: data/labels_v2/strict/{pair_labels.csv, sequences.csv.gz, new_sequences.fasta}, results/label_audit_v2.md
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import gzip
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

P = F2P_ROOT
E = P / "data/labels_external"
OUT = P / "data/labels_v2/strict"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_labels as bl  # noqa: E402
from fusondb_paediatric_coverage import PANEL, names  # noqa: E402

FOCUSED_MAX_CASES = 100
RT_GAP = 200_000


# ------------------------------------------------------------------ symbols
def hgnc_map():
    h = pd.read_csv(E / "hgnc/hgnc_complete_set.txt", sep="\t", dtype=str, usecols=["symbol", "prev_symbol", "alias_symbol", "status"])
    h = h[h.status == "Approved"]
    approved = set(h.symbol.str.upper())
    prev, alias = defaultdict(set), defaultdict(set)
    for s, p, a in zip(h.symbol.str.upper(), h.prev_symbol.fillna(""), h.alias_symbol.fillna("")):
        for x in p.split("|"):
            if x:
                prev[x.upper()].add(s)
        for x in a.split("|"):
            if x:
                alias[x.upper()].add(s)
    m = {}
    for d in (alias, prev):  # previous symbols override aliases; only unambiguous ones
        for k, v in d.items():
            if k not in approved and len(v) == 1:
                m[k] = next(iter(v))
    return m


HG = hgnc_map()


def norm(s):
    s = str(s).strip().upper()
    return HG.get(s, s)


def pkey(h, t):
    return f"{norm(h)}::{norm(t)}"


# ------------------------------------------------------------------ evidence
def mitelman_split():
    m = pd.read_csv(E / "mitelman/extracted/MBCA.TXT.DATA", sep="\t", dtype=str, encoding="latin-1")
    m = m[m.GeneShort.str.contains("::", na=False)]
    size = m.groupby("RefNo").size()
    focused = set(size.index[size <= FOCUSED_MAX_CASES])
    allc, foc = defaultdict(set), defaultdict(set)
    for ref, inv, g in zip(m.RefNo, m.InvNo, m.GeneShort):
        for item in g.split(","):
            parts = [x.strip() for x in item.split("::")]
            if len(parts) < 2 or any(not x or x.endswith("+") for x in parts):
                continue
            for h, t in zip(parts[:-1], parts[1:]):
                k = pkey(h, t)
                allc[k].add((ref, inv))
                if ref in focused:
                    foc[k].add((ref, inv))
    return {k: len(v) for k, v in allc.items()}, {k: len(v) for k, v in foc.items()}


def evidence():
    bl.sym = norm  # build_labels.key() now uses HGNC-normalised symbols
    civ_best, civ_any = bl.civic()
    ct = bl.ctat()
    sg = bl.singh()
    mit_all, mit_foc = mitelman_split()
    pairs = set(civ_any) | set(mit_all) | set(ct) | set(sg)
    d = pd.DataFrame(index=sorted(pairs))
    d.index.name = "pair"
    d["civic_best_level"] = [civ_best.get(k, "") for k in d.index]
    d["civic_any"] = d.index.isin(list(civ_any))
    d["chimerkb"] = ["ChimerKB" in ct.get(k, ()) for k in d.index]
    d["mitelman_cases_all"] = [mit_all.get(k, 0) for k in d.index]
    d["mitelman_cases_focused"] = [mit_foc.get(k, 0) for k in d.index]
    d["ctat_normal"] = [";".join(sorted(ct.get(k, set()) & set(bl.NORMAL_CTAT))) for k in d.index]
    d["ctat_cancer_any"] = [";".join(sorted(ct.get(k, set()) & set(bl.CANCER_ANY_CTAT))) for k in d.index]
    d["singh_s3"] = d.index.isin(list(sg))
    d["tier1"] = d.civic_best_level.isin(list("ABC")) | d.chimerkb
    d["tier2"] = d.mitelman_cases_focused >= 2
    d["positive"] = d.tier1 | d.tier2
    d["positive_ge3"] = d.tier1 | (d.mitelman_cases_focused >= 3)
    d["normal"] = (d.ctat_normal != "") | d.singh_s3
    d["cancer_any"] = d.civic_any | (d.ctat_cancer_any != "") | (d.mitelman_cases_all >= 1)
    ca = set(d.index[d.cancer_any])
    d["cancer_any_either"] = [k in ca or bl.rev(k) in ca for k in d.index]
    d["negative"] = d.normal & ~d.cancer_any_either
    d["label"] = np.where(d.positive, 1, np.where(d.negative, 0, -1))
    return d


def gene_coords():
    g = {}
    att = re.compile(r'gene_name "([^"]+)"')
    with gzip.open(E / "gencode_v50/gencode.v50.basic.annotation.gtf.gz", "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.split("\t", 9)
            if c[2] != "gene":
                continue
            n = norm(att.search(c[8]).group(1))
            if n not in g:  # first (primary assembly) entry wins
                g[n] = (c[0], int(c[3]), int(c[4]), c[6])
    return g


def pair_geometry(pairs, gc):
    out = {}
    for k in pairs:
        h, t = k.split("::", 1)
        if h not in gc or t not in gc:
            out[k] = (np.nan, None)
            continue
        (c1, s1, e1, st1), (c2, s2, e2, st2) = gc[h], gc[t]
        if c1 != c2:
            out[k] = (np.inf, False)
            continue
        gap = max(0, max(s1, s2) - min(e1, e2))
        up = (s1 < s2) if st1 == "+" else (e1 > e2)
        out[k] = (gap, bool(st1 == st2 and up and gap < RT_GAP))
    return out


# ------------------------------------------------------------------ sequences
def seq_sha(s):
    return hashlib.sha1(str(s).replace("*", "").upper().encode()).hexdigest()


def sequences():
    key = ["chr1", "bp1", "strand1", "chr2", "bp2", "strand2"]
    tr = pd.read_csv(P / "data/labels_v2/rebuilt_sequences.csv.gz")
    tr = tr[tr.frame == "in_frame"]
    tr["pair"] = [pkey(h, t) for h, t in zip(tr.head_gene, tr.tail_gene)]
    a = tr.assign(set="A_pipeline", origin=None)[["pair", "aa_seq", "head_aa", "set", "head_tx", "tail_tx"] + key]
    # which sources each breakpoint came from
    ch = __import__("translate_breakpoints").chimeras()[key + ["source"]].drop_duplicates()
    src = ch.groupby(key).source.agg(lambda x: ";".join(sorted(set(x)))).rename("bp_sources")
    a = a.join(src, on=key)
    fp = pd.read_csv(P / "data/fusionpath_rebuild/01_source_data/merged/merged_fusions.filtered.hg38.formatted.labeled.csv",
                     usecols=["hgene", "tgene", "aa_seq"], low_memory=False).dropna()
    b1 = pd.DataFrame({"pair": [pkey(h, t) for h, t in zip(fp.hgene, fp.tgene)], "aa_seq": fp.aa_seq, "bp_sources": "fusionpath_db"})
    f = pd.read_csv(P / "data/raw/fuson_db/FusOn-DB.csv", usecols=["fusiongenes", "aa_seq"])
    f = f.assign(pair=f.fusiongenes.str.split(",")).explode("pair")
    b2 = pd.DataFrame({"pair": [pkey(*p.split("::", 1)) if "::" in str(p) else None for p in f.pair], "aa_seq": f.aa_seq, "bp_sources": "fuson_db"})
    g = next((P / "data/raw/fusiongdb2").glob("combined*.txt"))
    gd = pd.read_csv(g, sep="\t", dtype=str, header=None, usecols=[7, 11, 18], low_memory=False).dropna()
    b3 = pd.DataFrame({"pair": [pkey(h, t) for h, t in zip(gd[7], gd[11])], "aa_seq": gd[18], "bp_sources": "fusiongdb2"})
    b = pd.concat([b1, b2, b3]).dropna(subset=["pair", "aa_seq"]).assign(set="B_existing")
    s = pd.concat([a, b], ignore_index=True)
    s["aa_seq"] = s.aa_seq.astype(str).str.replace("*", "", regex=False).str.upper()
    s = s[s.aa_seq.str.len() >= 20]
    s["sha1"] = [seq_sha(x) for x in s.aa_seq]
    return s


def folds(lp, seed, p=(0.70, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    genes = np.array(sorted(set(lp.hgene) | set(lp.tgene)))
    a = dict(zip(genes, rng.choice(3, size=len(genes), p=p)))
    h, t = lp.hgene.map(a), lp.tgene.map(a)
    both = lp[(h == 2) & (t == 2)]
    one = lp[((h == 2) | (t == 2))]
    return both, one


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = evidence()
    gc = gene_coords()
    seqs = sequences()
    allpairs = set(ev.index) | set(seqs.pair)
    geo = pair_geometry(allpairs, gc)
    ev["gene_gap_bp"] = [geo[k][0] for k in ev.index]
    ev["readthrough"] = [geo[k][1] for k in ev.index]
    ev.to_csv(OUT / "pair_labels.csv")
    seqs = seqs.join(ev[["label", "tier1", "tier2", "positive_ge3", "readthrough", "gene_gap_bp"]], on="pair")
    seqs["label"] = seqs.label.fillna(-1).astype(int)
    panel_pairs = {pkey(hh, tt) for h, t, _ in PANEL for hh in names(h) for tt in names(t)}
    seqs["panel"] = seqs.pair.isin(panel_pairs)
    seqs["hgene"] = seqs.pair.str.split("::").str[0]
    seqs["tgene"] = seqs.pair.str.split("::").str[1]
    emb = set(pd.read_csv(P / "data/embeddings/esm2_650m/index.csv").seq_sha1)
    seqs["embedded"] = seqs.sha1.isin(emb)
    # a sequence carrying two different labels (several pairs) is dropped
    lab = seqs[(seqs.label >= 0) & ~seqs.panel]
    conflict = lab.groupby("sha1").label.nunique()
    lab = lab[~lab.sha1.isin(conflict.index[conflict > 1])]
    seqs.drop(columns=["aa_seq"]).to_csv(OUT / "sequences.csv.gz", index=False)
    new = lab[~lab.embedded].drop_duplicates("sha1")
    with open(OUT / "new_sequences.fasta", "w") as fh:
        for sh, aa in zip(new.sha1, new.aa_seq):
            fh.write(f">{sh}\n{aa}\n")

    roles = pd.read_csv(P / "data/fusionpath_rebuild/01_source_data/merged/merged_fusions.filtered.hg38.formatted.labeled.csv",
                        usecols=["hgene", "tgene", "hgene_role", "tgene_role"], low_memory=False)
    role_genes = {norm(g) for g in pd.concat([roles.hgene[roles.hgene_role.notna()], roles.tgene[roles.tgene_role.notna()]])}

    T = ["# Label set v2, strict (HGNC-normalised, focused Mitelman, single translation pipeline)\n",
         f"HGNC map: {len(HG):,} previous/alias symbols mapped to approved symbols (unambiguous only).\n",
         "## 1. Pair-level evidence\n"]
    T.append(pd.Series({"pairs in any source": len(ev), "tier 1": int(ev.tier1.sum()), "tier 2 (focused Mitelman >= 2)": int(ev.tier2.sum()),
                        "positive": int(ev.positive.sum()), "positive (Mitelman >= 3 variant)": int(ev.positive_ge3.sum()),
                        "normal-tissue recurrent": int(ev.normal.sum()), "negative": int(ev.negative.sum()),
                        "negative, not read-through": int((ev.negative & (ev.readthrough != True)).sum()),  # noqa: E712
                        "normal and positive (positive wins)": int((ev.normal & ev.positive).sum())}).to_frame("pairs").to_markdown() + "\n")
    T.append("## 2. Labelled sequences (panel pairs removed; sequences with conflicting labels dropped)\n")
    for name, sub in [("A: single pipeline (GENCODE, canonical junctions)", lab[lab.set == "A_pipeline"]),
                      ("A + B: plus existing database sequences", lab)]:
        u = sub.drop_duplicates("sha1")
        up = sub.drop_duplicates("pair")
        T.append(f"**{name}**\n")
        tab = pd.DataFrame({"pairs": up.groupby("label").size(),
                            "pairs, not read-through": up[up.readthrough != True].groupby("label").size(),  # noqa: E712
                            "unique sequences": u.groupby("label").size(),
                            "already embedded": u[u.embedded].groupby("label").size(),
                            "new (need embedding)": u[~u.embedded].groupby("label").size()}).fillna(0).astype(int)
        T.append(tab.rename(index={0: "negative", 1: "positive"}).to_markdown() + "\n")
        if name.startswith("A:"):
            T.append("Breakpoint sources of set-A sequences: " +
                     str(sub.groupby(["label", "bp_sources"]).sha1.nunique().to_dict()) + "\n")
    T.append(f"Sequences needing embedding (A + B, labelled): {len(new):,} -> {OUT / 'new_sequences.fasta'}\n")

    T.append("## 3. Possible shortcuts\n")
    up = lab.drop_duplicates("pair").copy()
    up["role_any"] = up.hgene.isin(role_genes) | up.tgene.isin(role_genes)
    T.append("Share of labelled pairs with an oncogene/TSG partner (FusionPath's rule), and read-through share:\n")
    T.append(up.groupby("label").agg(pairs=("pair", "size"), role_any=("role_any", "mean"),
                                     readthrough=("readthrough", lambda x: (x == True).mean()),  # noqa: E712
                                     interchromosomal=("gene_gap_bp", lambda x: np.isinf(x).mean())).round(3)
             .rename(index={0: "negative", 1: "positive"}).to_markdown() + "\n")
    L = lab.drop_duplicates("sha1")
    T.append("Protein length by label and set (median, IQR):\n")
    T.append(L.assign(n=L.aa_seq.str.len()).groupby(["set", "label"]).n.describe(percentiles=[.25, .5, .75])[["count", "25%", "50%", "75%"]]
             .round(0).to_markdown() + "\n")

    T.append("## 4. Gene-disjoint power (11 gene assignments, 0.70/0.15/0.15)\n")
    for name, sub in [("A", lab[lab.set == "A_pipeline"]), ("A+B", lab)]:
        lp = sub.drop_duplicates("pair")[["pair", "hgene", "tgene", "label", "readthrough"]]
        rows = []
        for seed in [42, *range(1, 11)]:
            both, one = folds(lp, seed)
            rows.append({"seed": seed, "both_unseen_pos": int((both.label == 1).sum()), "both_unseen_neg": int((both.label == 0).sum()),
                         "both_unseen_neg_non_rt": int(((both.label == 0) & (both.readthrough != True)).sum()),  # noqa: E712
                         "one_unseen_pos": int((one.label == 1).sum()), "one_unseen_neg": int((one.label == 0).sum())})
        gd = pd.DataFrame(rows)
        T.append(f"**{name}** (pairs): min/median both-unseen positives {gd.both_unseen_pos.min()}/{int(gd.both_unseen_pos.median())}, "
                 f"negatives {gd.both_unseen_neg.min()}/{int(gd.both_unseen_neg.median())}; at-least-one-unseen positives "
                 f"{gd.one_unseen_pos.min()}/{int(gd.one_unseen_pos.median())}, negatives {gd.one_unseen_neg.min()}/{int(gd.one_unseen_neg.median())}\n")
        T.append(gd.to_markdown(index=False) + "\n")
    T.append("## 5. Old FusionPath labels of the new labelled pairs\n")
    old = pd.read_csv(P / "data/labels_v2/row_labels.csv.gz", usecols=["pair", "cancer_label"])
    old["pair"] = [pkey(*p.split("::", 1)) for p in old.pair]
    ol = old.groupby("pair").cancer_label.agg(lambda x: "/".join(sorted(set(x))))
    up["old"] = up.pair.map(ol).fillna("not in FusionPath")
    T.append(pd.crosstab(up.old, up.label.map({0: "negative", 1: "positive"}), margins=True).to_markdown() + "\n")
    (P / "results/label_audit_v2.md").write_text("\n".join(T))
    print("\n".join(T))


if __name__ == "__main__":
    main()
