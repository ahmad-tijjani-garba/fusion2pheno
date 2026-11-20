"""Revision: provenance for numbers the reviewers could not find in any result file (read-only analysis).

Writes results/provenance.md and results/provenance.csv with, for each number, its value and how it was computed:
  125-gene role list; normal-tissue records missing from the rebuild; 21,065 cases in the largest Mitelman reference;
  GENCODE v50 transcript counts; DNAJB1-PRKACA and CBFB-MYH11 gene distances; KMT2A-headed fusions in FusOn-pLM's
  training split; KMT2A records labelled Benign in the rebuild; Singh et al. gene agreement.
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import gzip
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

P = F2P_ROOT
O = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_labels_v2 import norm  # noqa: E402

rows = []


def add(item, value, how):
    rows.append({"item": item, "value": value, "how": how})
    print(f"{item}: {value}", flush=True)


M = P / "data/fusionpath_rebuild/01_source_data/merged/merged_fusions.filtered.hg38.formatted.labeled.csv"
d = pd.read_csv(M, low_memory=False, usecols=["hgene", "tgene", "hgene_role", "tgene_role", "cancer_label"])
genes = pd.concat([d.loc[d.hgene_role.notna(), "hgene"], d.loc[d.tgene_role.notna(), "tgene"]]).astype(str).str.upper()
add("genes with an oncogene/TSG role in the rebuilt FusionPath data", genes.nunique(),
    "unique head/tail symbols with a non-empty hgene_role/tgene_role in the as-published rebuild (role column used by the released code)")
add("labelled records in the rebuild (as published)", int(d.cancer_label.isin(["Benign", "Pathogenic"]).sum()), "Benign + Pathogenic rows")
add("normal-tissue candidates in FusionPath's dataset", 5403,
    "stated in the FusionPath preprint (Kurz et al., 2026, Methods): STAR-Fusion candidates from 27 normal-tissue RNA-seq sets; not rebuilt")

E = P / "data/labels_external"
m = pd.read_csv(E / "mitelman/extracted/MBCA.TXT.DATA", sep="\t", dtype=str, encoding="latin-1")
m = m[m.GeneShort.str.contains("::", na=False)]
size = m.groupby("RefNo").size().sort_values(ascending=False)
add("fusion cases in the largest Mitelman reference", int(size.iloc[0]),
    f"MBCA.TXT.DATA rows with a gene fusion (GeneShort contains '::'), RefNo {size.index[0]}")
add("Mitelman references with <=100 fusion cases (focused)", int((size <= 100).sum()), f"of {len(size)} references reporting fusions")

G = P / "data/labels_external/gencode_v50"
n_fa, n_cds = 0, 0
with gzip.open(G / "gencode.v50.pc_transcripts.fa.gz", "rt") as f:
    for line in f:
        if line.startswith(">"):
            n_fa += 1
            n_cds += "CDS:" in line
basic_tx, coords = set(), {}
att = re.compile(r'(\S+) "([^"]*)"')
with gzip.open(G / "gencode.v50.basic.annotation.gtf.gz", "rt") as f:
    for line in f:
        if line.startswith("#"):
            continue
        c = line.split("\t", 9)
        if c[2] == "transcript":
            a = dict(att.findall(c[8]))
            if a.get("transcript_type") == "protein_coding":
                basic_tx.add(a["transcript_id"])
        elif c[2] == "gene":
            g = dict(att.findall(c[8])).get("gene_name")
            if g and g not in coords:
                coords[g] = (c[0], int(c[3]), int(c[4]), c[6])
add("GENCODE v50 protein-coding transcript sequences (pc_transcripts FASTA)", n_fa, f"{n_cds} with an annotated CDS")
add("protein-coding transcripts in GENCODE v50 basic", len(basic_tx), "transcript_type protein_coding in the basic GTF (exon structures used for translation)")
for h, t in [("DNAJB1", "PRKACA"), ("CBFB", "MYH11"), ("KIAA1549", "BRAF")]:
    (c1, s1, e1, st1), (c2, s2, e2, st2) = coords[h], coords[t]
    gap = max(0, max(s1, s2) - min(e1, e2)) if c1 == c2 else None
    add(f"{h}-{t} gene layout", f"{c1} {st1} / {c2} {st2}; gap {gap:,} bp" if gap is not None else f"{c1} / {c2}",
        "GENCODE v50 basic gene coordinates")

SNAP = next((P / "models/hf_cache/hub/models--ChatterjeeLab--FusOn-pLM/snapshots").iterdir())
tr = pd.read_csv(SNAP / "fuson_plm/data/splits/train_df.csv", usecols=["sequence"])
db = pd.read_csv(SNAP / "fuson_plm/data/fuson_db.csv")
seqcol = "aa_seq" if "aa_seq" in db.columns else "sequence"
db = db[db[seqcol].isin(set(tr.sequence))]
pairs = set()
for fg in db.fusiongenes.astype(str):
    for p in re.split(r"[,;|]", fg):
        if "::" in p:
            a, b = (x.strip().upper() for x in p.split("::", 1))
            pairs.add((norm(a), norm(b)))
kmt = sorted(b for a, b in pairs if a == "KMT2A")
add("KMT2A-headed fusion pairs in FusOn-pLM's training split", len(kmt), "unique KMT2A::partner pairs (HGNC-normalised) among training sequences: " + ", ".join(kmt))

rb = (O / "results/rebuild_summary.md").read_text()
add("panel records in the rebuild (Non-Cancer fix applied)", re.search(r"records: (\d+)", rb).group(1), "results/rebuild_summary.md")
km = [l for l in rb.splitlines() if "KMT2A::" in l and "|" in l]
add("KMT2A rows in the rebuilt panel table", " ; ".join(" ".join(x.split()) for x in km), "results/rebuild_summary.md (columns: records, Pathogenic, Benign, Uncertain)")
tb = (O / "results/translate_breakpoints.md").read_text()
add("Singh et al. gene agreement (final run)", re.search(r"EnsemblGene1 in ([\d.]+%).*tail ([\d.]+%)", tb).group(0), "results/translate_breakpoints.md")

out = pd.DataFrame(rows)
(P / "results").mkdir(exist_ok=True)
out.to_csv(P / "results/provenance.csv", index=False)
(P / "results/provenance.md").write_text("# Provenance of numbers not in other result files\n\n" + out.to_markdown(index=False) + "\n")
print("written")
