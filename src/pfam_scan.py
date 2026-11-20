"""Pfam domain calls (HMMER via pyhmmer, Pfam-A 39.0, gathering thresholds) for every sequence the revision needs:
FusionPath-style labelled fusion sequences (as published), gene-role-free sequences, and MANE wild-type partner proteins.
Stands in for FusionPath's InterProScan step (Pfam member only). Outputs data/domains/{pfam_hits.csv.gz (seq_sha1, set, pfam_acc, pfam_name, env_from, env_to, score),
pfam_scanned.csv (seq_sha1, set, length)}.
"""
from f2p_root import F2P_CACHE, F2P_ROOT  # paths: see src/f2p_root.py
import gzip
import hashlib
import sys
import time
from pathlib import Path

import pandas as pd
import pyhmmer

P = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
PF = F2P_CACHE / "pfam"
OUT = P / "data/domains"
OUT.mkdir(parents=True, exist_ok=True)


def sha(s):
    return hashlib.sha1(s.encode()).hexdigest()


def txt(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else x


def clean(s):
    return str(s).replace("*", "").strip().upper()


seqs = {}  # sha1 -> (seq, set)
import fp_benchmark_rev as FB  # noqa: E402
for s in FB.load("as_published").seq:
    seqs.setdefault(sha(s), (s, "fusionpath_style"))
import train_v2 as T  # noqa: E402
import probes_v2 as PR  # noqa: E402
s2, _ = T.load()
for h, s in PR.sequences(s2).items():
    seqs.setdefault(h, (clean(s), "gene_role_free"))
wt = pd.read_csv(P / "external/fusionpath/fusionpath/data/gene_wt_mane_sequences.csv").dropna()
for s in wt.sequence:
    s = FB.seq_norm(s)
    seqs.setdefault(sha(s), (s, "wild_type"))
seqs = {k: v for k, v in seqs.items() if len(v[0]) >= 10}
print(f"{len(seqs):,} sequences, {sum(len(v[0]) for v in seqs.values()):,} residues", flush=True)
pd.DataFrame([(k, v[1], len(v[0])) for k, v in seqs.items()], columns=["seq_sha1", "set", "length"]).to_csv(
    OUT / "pfam_scanned.csv", index=False)

alpha = pyhmmer.easel.Alphabet.amino()
valid = set("ACDEFGHIKLMNPQRSTVWY")
block = pyhmmer.easel.TextSequenceBlock(
    [pyhmmer.easel.TextSequence(name=k, sequence="".join(c if c in valid else "X" for c in v[0]))
     for k, v in seqs.items()]).digitize(alpha)
setof = {k: v[1] for k, v in seqs.items()}
t0 = time.time()
rows = []
with pyhmmer.plan7.HMMFile(gzip.open(PF / "Pfam-A.hmm.gz")) as hf:
    for n, hits in enumerate(pyhmmer.hmmsearch(hf, block, cpus=int(__import__("os").environ.get("SLURM_CPUS_PER_TASK", 8)),
                                               bit_cutoffs="gathering")):
        acc = txt(hits.query.accession) or ""
        name = txt(hits.query.name)
        for hit in hits.included:
            for dom in hit.domains.included:
                k = txt(hit.name)
                rows.append((k, setof[k], acc.split(".")[0], name, dom.env_from, dom.env_to, round(dom.score, 1)))
        if n % 5000 == 0:
            print(f"  {n:,} families, {len(rows):,} domain hits, {time.time() - t0:.0f}s", flush=True)
h = pd.DataFrame(rows, columns=["seq_sha1", "set", "pfam_acc", "pfam_name", "env_from", "env_to", "score"])
h.to_csv(OUT / "pfam_hits.csv.gz", index=False)
print(f"done: {len(h):,} domain hits on {h.seq_sha1.nunique():,} of {len(seqs):,} sequences, {time.time() - t0:.0f}s")
print(h.groupby("set").seq_sha1.nunique())
