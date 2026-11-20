"""Translate fusion breakpoints (hg38) into fusion proteins with GENCODE v50 transcripts.

For each chimera (chr1, bp1, strand1, chr2, bp2, strand2):
  head transcripts: protein-coding transcripts with an exon whose 3' end (transcript direction) is bp1
  tail transcripts: protein-coding transcripts with an exon whose 5' start (transcript direction) is bp2
  fusion = head CDS up to the junction + tail transcript from the junction; translated to the first stop.
  frame: in_frame when the junction falls inside both CDSs with matching codon phase; otherwise
  frameshift / head_not_coding / tail_5utr / tail_3utr / head_past_stop.
Sources: Singh 2020 Table S3 (GTEx normal tissue) and the FusionPath merged table (hg38 breakpoints, used both
as a validation of this translator against FusionPath's own aa_seq and to put positives through the same pipeline).
Also writes the junction position in the protein (head_aa = number of head residues).

Outputs: data/labels_v2/rebuilt_sequences.csv.gz (one row per chimera x transcript pair, in-frame and not),
         results/translate_breakpoints.md
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import gzip
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

P = F2P_ROOT
G = P / "data/labels_external/gencode_v50"
OUT = P / "data/labels_v2"

BASES = "TCAG"
CODON = {a + b + c: aa for (a, b, c), aa in zip(
    [(a, b, c) for a in BASES for b in BASES for c in BASES],
    "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")}


def translate(nt):
    out = []
    for i in range(0, len(nt) - 2, 3):
        aa = CODON.get(nt[i:i + 3], "X")
        if aa == "*":
            return "".join(out), True
        out.append(aa)
    return "".join(out), False


def load_transcripts():
    seqs, cds = {}, {}
    with gzip.open(G / "gencode.v50.pc_transcripts.fa.gz", "rt") as f:
        tid, buf = None, []
        for line in f:
            if line.startswith(">"):
                if tid:
                    seqs[tid] = "".join(buf)
                h = line[1:].rstrip().split("|")
                tid, buf = h[0], []
                c = next((x for x in h if x.startswith("CDS:")), None)
                if c:
                    s, e = c[4:].split("-")
                    cds[tid] = (int(s), int(e))
            else:
                buf.append(line.strip().upper())
        if tid:
            seqs[tid] = "".join(buf)
    exons = defaultdict(list)
    meta = {}
    att = re.compile(r'(\S+) "([^"]*)"')
    with gzip.open(G / "gencode.v50.basic.annotation.gtf.gz", "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if c[2] != "exon":
                continue
            a = defaultdict(list)
            for k, v in att.findall(c[8]):
                a[k].append(v)
            tid = a["transcript_id"][0]
            if tid not in cds:
                continue
            exons[tid].append((int(c[3]), int(c[4])))
            if tid not in meta:
                meta[tid] = {"chrom": c[0], "strand": c[6], "gene_id": a["gene_id"][0].split(".")[0],
                             "gene_name": a["gene_name"][0], "mane": "MANE_Select" in a["tag"],
                             "canonical": "Ensembl_canonical" in a["tag"]}
    donors, acceptors, tx = defaultdict(list), defaultdict(list), {}
    bad = 0
    for tid, ex in exons.items():
        m = meta[tid]
        ex = sorted(ex, reverse=(m["strand"] == "-"))
        if sum(e - s + 1 for s, e in ex) != len(seqs.get(tid, "")):
            bad += 1
            continue
        pos = 0
        for s, e in ex:
            t_start, t_end = pos + 1, pos + (e - s + 1)
            five, three = (s, e) if m["strand"] == "+" else (e, s)
            donors[(m["chrom"], m["strand"], three)].append((tid, t_end))
            acceptors[(m["chrom"], m["strand"], five)].append((tid, t_start))
            pos = t_end
        tx[tid] = m
    print(f"GENCODE v50: {len(tx):,} protein-coding transcripts usable ({bad} skipped: length mismatch)", flush=True)
    return seqs, cds, tx, donors, acceptors


def fuse(seqs, cds, h, th, t, tt):
    hcs, hce = cds[h]
    tcs, tce = cds[t]
    if th < hcs:
        return "head_not_coding", "", 0
    if th > hce:
        return "head_past_stop", "", 0
    head_len = th - hcs + 1
    nt = seqs[h][hcs - 1:th] + seqs[t][tt - 1:]
    aa, stopped = translate(nt)
    if tt < tcs:
        cls = "tail_5utr"
    elif tt > tce:
        cls = "tail_3utr"
    elif head_len % 3 == (tt - tcs) % 3:
        cls = "in_frame"
    else:
        cls = "frameshift"
    if cls == "in_frame" and not stopped:
        cls = "in_frame_no_stop"
    return cls, aa, head_len // 3


def chimeras():
    s = pd.read_excel(P / "data/labels_external/singh2020/extracted/supp/Table S.xlsx", sheet_name="S3", header=None, dtype=str)
    h = next(i for i in range(40) if (s.iloc[i].astype(str) == "GeneName1").any())
    s.columns = s.iloc[h].astype(str)
    s = s.iloc[h + 1:].dropna(subset=["GeneName1", "Breakpoint1"])
    a = pd.DataFrame({"source": "singh2020", "source_row": s.index, "gene1": s.GeneName1, "gene2": s.GeneName2,
                      "ens1": s.EnsemblGene1, "ens2": s.EnsemblGene2, "fusiontype": s.fusiontype,
                      "chr1": "chr" + s.chr1.astype(str).str.replace("chr", ""), "bp1": s.Breakpoint1.astype(int),
                      "strand1": s.strand1, "chr2": "chr" + s.chr2.astype(str).str.replace("chr", ""),
                      "bp2": s.Breakpoint2.astype(int), "strand2": s.strand2, "given_aa": None})
    f = pd.read_csv(P / "data/fusionpath_rebuild/01_source_data/merged/merged_fusions.filtered.hg38.formatted.labeled.csv",
                    usecols=["hgene", "tgene", "hgene_chr", "hgene_bp", "hgene_strand", "tgene_chr", "tgene_bp",
                             "tgene_strand", "aa_seq"], low_memory=False)
    f["source_row"] = np.arange(len(f))
    f = f.dropna(subset=["hgene_chr", "hgene_bp", "tgene_chr", "tgene_bp"])
    b = pd.DataFrame({"source": "fusionpath", "source_row": f.source_row, "gene1": f.hgene, "gene2": f.tgene,
                      "ens1": None, "ens2": None, "fusiontype": None, "chr1": f.hgene_chr, "bp1": f.hgene_bp.astype(int),
                      "strand1": f.hgene_strand, "chr2": f.tgene_chr, "bp2": f.tgene_bp.astype(int),
                      "strand2": f.tgene_strand, "given_aa": f.aa_seq})
    return pd.concat([a, b], ignore_index=True)


def main():
    seqs, cds, tx, donors, acceptors = load_transcripts()
    ch = chimeras()
    key = ["chr1", "bp1", "strand1", "chr2", "bp2", "strand2"]
    uniq = ch[key].drop_duplicates().reset_index(drop=True)
    print(f"chimeras: {len(ch):,} rows, {len(uniq):,} unique breakpoint pairs", flush=True)
    diag = defaultdict(int)
    rows, uniq_hash = [], {}
    for r in uniq.itertuples(index=False):
        hs = donors.get((r.chr1, r.strand1, r.bp1), [])
        ts = acceptors.get((r.chr2, r.strand2, r.bp2), [])
        for off in (-1, 1):  # off-by-one diagnostic only
            diag[f"head_match_offset{off:+d}"] += bool(donors.get((r.chr1, r.strand1, r.bp1 + off)))
            diag[f"tail_match_offset{off:+d}"] += bool(acceptors.get((r.chr2, r.strand2, r.bp2 + off)))
        diag["head_match"] += bool(hs)
        diag["tail_match"] += bool(ts)
        diag["both_match"] += bool(hs and ts)
        best, classes, n_inframe_uniq = None, defaultdict(int), set()
        for h, th in hs:
            for t, tt in ts:
                cls, aa, head_aa = fuse(seqs, cds, h, th, t, tt)
                classes[cls] += 1
                if cls != "in_frame":
                    continue
                n_inframe_uniq.add(aa)
                rank = (tx[h]["mane"] and tx[t]["mane"], tx[h]["canonical"] and tx[t]["canonical"], len(aa), h + t)
                if best is None or rank > best[0]:
                    best = (rank, h, t, aa, head_aa)
        if not (hs and ts):
            continue
        uniq_hash[tuple(r)] = {hash(a) for a in n_inframe_uniq}
        frame = "in_frame" if best else max(classes, key=classes.get)
        h, t, aa, head_aa = (best[1:] if best else (hs[0][0], ts[0][0], "", 0))
        rows.append((r.chr1, r.bp1, r.strand1, r.chr2, r.bp2, r.strand2, h, t, tx[h]["gene_name"], tx[t]["gene_name"],
                     tx[h]["gene_id"], tx[t]["gene_id"], bool(best and best[0][0]), bool(best and best[0][1]),
                     frame, aa, head_aa, len(aa), len(n_inframe_uniq), sum(classes.values())))
    tr = pd.DataFrame(rows, columns=key + ["head_tx", "tail_tx", "head_gene", "tail_gene", "head_ens", "tail_ens",
                                           "both_mane", "both_canonical", "frame", "aa_seq", "head_aa", "aa_len",
                                           "n_unique_inframe", "n_tx_pairs"])
    OUT.mkdir(parents=True, exist_ok=True)
    tr.to_csv(OUT / "rebuilt_sequences.csv.gz", index=False)

    T = ["# Breakpoint translation with GENCODE v50\n",
         f"Unique breakpoint pairs: {len(uniq):,}. Exon-boundary matches (exact): head {diag['head_match']:,}, "
         f"tail {diag['tail_match']:,}, both {diag['both_match']:,}. Off-by-one controls: head -1 {diag['head_match_offset-1']:,}, "
         f"+1 {diag['head_match_offset+1']:,}; tail -1 {diag['tail_match_offset-1']:,}, +1 {diag['tail_match_offset+1']:,}.\n"]
    m = ch.merge(tr, on=key, how="left")
    per = m.groupby(["source", "source_row"]).agg(matched=("frame", lambda x: x.notna().any()),
                                                 in_frame=("frame", lambda x: (x == "in_frame").any()))
    T.append("Per input row:\n")
    T.append(per.groupby("source").agg(rows=("matched", "size"), both_ends_on_exon_boundary=("matched", "sum"),
                                       any_in_frame=("in_frame", "sum")).to_markdown() + "\n")
    T.append("Frame class per input row (in_frame if any transcript pair is in frame, else the most common class):\n")
    T.append(m.dropna(subset=["frame"]).groupby(["source", "frame"]).size().unstack(0).fillna(0).astype(int).to_markdown() + "\n")
    T.append("Unique in-frame proteins per breakpoint (before picking one representative):\n")
    T.append(tr[tr.frame == "in_frame"].n_unique_inframe.describe(percentiles=[.5, .9, .99]).round(1).to_frame().T.to_markdown(index=False) + "\n")
    # validation against FusionPath's own sequences
    fp = m[(m.source == "fusionpath") & (m.frame == "in_frame")].copy()
    fp["given"] = fp.given_aa.astype(str).str.replace("*", "", regex=False).str.upper()
    any_hit = [hash(g) in uniq_hash.get(tuple(k), ()) for g, k in zip(fp.given, fp[key].itertuples(index=False))]
    rep_hit = (fp.given == fp.aa_seq)
    T.append(f"Validation: of {len(fp):,} FusionPath rows with an in-frame translation, {sum(any_hit):,} "
             f"({np.mean(any_hit):.1%}) have FusionPath's own aa_seq among our unique in-frame products, and "
             f"{int(rep_hit.sum()):,} ({rep_hit.mean():.1%}) equal our single representative.\n")
    T.append("Representative choice: MANE pair > Ensembl-canonical pair > longest product.\n")
    gm = m.dropna(subset=["frame"])
    gm = gm[gm.source == "singh2020"]
    if len(gm):
        T.append(f"Singh gene check: head Ensembl gene agrees with EnsemblGene1 in {(gm.head_ens == gm.ens1).mean():.1%} "
                 f"of matched products; tail {(gm.tail_ens == gm.ens2).mean():.1%}.\n")
    (P / "results/translate_breakpoints.md").write_text("\n".join(T))
    print("\n".join(T))


if __name__ == "__main__":
    main()
