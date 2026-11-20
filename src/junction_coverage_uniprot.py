"""How much does UniProt (Swiss-Prot human, canonical + isoforms) raise junction coverage over MANE alone?

No embedding or training: sequence matching only. Uses the same sequence set and matching rule (15-aa prefix /
suffix, extended) as embed_pooled.py, but tries every UniProt isoform of the head/tail gene and keeps the longest
match. Writes data/junctions/junctions_mane_uniprot.csv (for a later re-pooling run) and
results/junction_coverage_uniprot.md.
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import gzip
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

P = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_pooled import collect_sequences, find_junction  # noqa: E402

UNIPROT = P / "data/raw/uniprot/uniprot_human_sprot_isoforms.fasta.gz"


def read_uniprot():
    by_gene, gene, buf = defaultdict(list), None, []
    with gzip.open(UNIPROT, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if gene and buf:
                    by_gene[gene].append("".join(buf))
                m = re.search(r" GN=(\S+)", line)
                gene, buf = (m.group(1).upper() if m else None), []
            else:
                buf.append(line.strip())
    if gene and buf:
        by_gene[gene].append("".join(buf))
    return by_gene


def junction_multi(seq, hcands, tcands):
    he = [find_junction(seq, h, None)[0] for h in hcands]
    he = max([x for x in he if x is not None], default=None)
    ts = [find_junction(seq, None, t)[1] for t in tcands]
    ts = min([x for x in ts if x is not None], default=None)
    if he is not None and ts is not None:
        if ts < he:
            he = ts = (he + ts) // 2
        return he, ts, "both"
    if he is not None:
        return he, he, "head"
    if ts is not None:
        return ts, ts, "tail"
    return None, None, "none"


def main():
    wt = pd.read_csv(P / "external/fusionpath/fusionpath/data/gene_wt_mane_sequences.csv")
    mane = dict(zip(wt.gene.str.upper(), wt.sequence.astype(str).str.replace("*", "", regex=False)))
    up = read_uniprot()
    print(f"MANE genes {len(mane):,}; UniProt genes {len(up):,}, sequences {sum(map(len, up.values())):,}", flush=True)

    seq2genes = collect_sequences()
    panel = set(pd.read_csv(P / "results/panel_scores.csv").sha1)
    panel_fusion = dict(pd.read_csv(P / "results/panel_scores.csv")[["sha1", "fusion"]].values)
    rows = []
    for s, (h, t) in seq2genes.items():
        a = junction_multi(s, [mane[h]] if h in mane else [], [mane[t]] if t in mane else [])
        b = junction_multi(s, ([mane[h]] if h in mane else []) + up.get(h, []),
                           ([mane[t]] if t in mane else []) + up.get(t, []))
        sha = hashlib.sha1(s.encode()).hexdigest()
        rows.append({"seq_sha1": sha, "length": len(s), "hgene": h, "tgene": t,
                     "panel": sha in panel, "fusion": panel_fusion.get(sha),
                     "src_mane": a[2], "head_end": b[0], "tail_start": b[1], "src_mane_uniprot": b[2],
                     "h_known": h in mane or h in up, "t_known": t in mane or t in up})
    d = pd.DataFrame(rows)
    (P / "data/junctions").mkdir(parents=True, exist_ok=True)
    d.to_csv(P / "data/junctions/junctions_mane_uniprot.csv", index=False)

    def tab(x):
        out = pd.DataFrame({"MANE": x.src_mane.value_counts(), "MANE+UniProt": x.src_mane_uniprot.value_counts()})
        out = out.reindex(["both", "head", "tail", "none"]).fillna(0).astype(int)
        out.loc["any found (%)"] = [round(100 * (x[c] != "none").mean(), 1) for c in ("src_mane", "src_mane_uniprot")]
        out.loc["both found (%)"] = [round(100 * (x[c] == "both").mean(), 1) for c in ("src_mane", "src_mane_uniprot")]
        return out

    tr, pn = d[~d.panel], d[d.panel]
    none_b = tr[tr.src_mane_uniprot == "none"]
    why = pd.Series({"head and tail gene both unknown": int((~none_b.h_known & ~none_b.t_known).sum()),
                     "one gene unknown": int((none_b.h_known ^ none_b.t_known).sum()),
                     "both genes known, no 15-aa match": int((none_b.h_known & none_b.t_known).sum())})
    pf = pn.groupby("fusion").agg(isoforms=("seq_sha1", "size"),
                                  mane_any=("src_mane", lambda s: round((s != "none").mean(), 2)),
                                  uniprot_any=("src_mane_uniprot", lambda s: round((s != "none").mean(), 2)),
                                  mane_both=("src_mane", lambda s: round((s == "both").mean(), 2)),
                                  uniprot_both=("src_mane_uniprot", lambda s: round((s == "both").mean(), 2)))
    changed = pf[(pf.mane_any != pf.uniprot_any) | (pf.mane_both != pf.uniprot_both)]
    text = ["# Junction coverage: MANE vs MANE + UniProt (Swiss-Prot human, release 2026_04, with isoforms)", "",
            f"## Training/benchmark sequences (n = {len(tr):,})", "", tab(tr).to_markdown(), "",
            "Still no junction with MANE + UniProt, by reason:", "", why.to_frame("sequences").to_markdown(), "",
            f"## Paediatric panel isoforms (n = {len(pn):,})", "", tab(pn).to_markdown(), "",
            "Panel fusions whose coverage changed:", "",
            (changed.to_markdown() if len(changed) else "(none)"), ""]
    (P / "results/junction_coverage_uniprot.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
