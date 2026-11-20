"""Partner- and junction-aware pooled embeddings (inference only).

For every labelled sequence and every paediatric-panel isoform:
  1. locate the fusion junction by matching the start of the fusion to the head gene's
     wild-type protein and the end to the tail gene's (MANE sequences shipped with FusionPath);
  2. compute per-residue embeddings (same overlapping 1,022-residue windows as before);
  3. pool five ways: mean, max, head-part mean, tail-part mean, junction-window mean (+/-50 aa).
If no wild-type match is found, head/tail/junction pools fall back to the whole-sequence mean
(flagged in junctions.csv).

Outputs in data/embeddings_pooled/<tag>/: pools.npz (float16 arrays, one per pooling),
index.csv (seq_sha1, length, head_end, tail_start, junction_source, n_windows)
Usage: python embed_pooled.py <hf_model_id> <tag> [--batch-tokens 12000]
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

P = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_sequences import CHUNK, windows  # noqa: E402
from fusondb_paediatric_coverage import PANEL, names  # noqa: E402
from train_seq_models import norm_seq  # noqa: E402

M = P / "data/fusionpath_rebuild/01_source_data/merged"
FILES = ["merged_fusions.filtered.hg38.formatted.labeled.csv",
         "merged_fusions.filtered.hg38.formatted.noncancerfix.labeled.csv"]
JW = 50  # junction half-window
POOLS = ["mean", "max", "head", "tail", "junction"]


def collect_sequences():
    """sequence -> (hgene, tgene) for labelled records and panel isoforms."""
    seqs = {}
    for f in FILES:
        d = pd.read_csv(M / f, low_memory=False, usecols=["hgene", "tgene", "aa_seq", "cancer_label"])
        d = d[d.cancer_label.isin(["Benign", "Pathogenic"]) & d.aa_seq.notna()]
        for s, h, t in zip(d.aa_seq, d.hgene, d.tgene):
            seqs.setdefault(norm_seq(s), (str(h).upper(), str(t).upper()))
    full = pd.read_csv(M / FILES[0], low_memory=False, usecols=["hgene", "tgene", "aa_seq"])
    fdb = pd.read_csv(P / "data/raw/fuson_db/FusOn-DB.csv", usecols=["aa_seq", "fusiongenes"])
    parts = fdb.fusiongenes.str.split("::", n=1, expand=True)
    fdb["hgene"], fdb["tgene"] = parts[0], parts[1]
    for df in (full, fdb):
        hg, tg = df.hgene.astype(str).str.upper(), df.tgene.astype(str).str.upper()
        for h, t, _ in PANEL:
            m = hg.isin(names(h)) & tg.isin(names(t)) & df.aa_seq.notna()
            for s in df.aa_seq[m]:
                seqs.setdefault(norm_seq(s), (h, t))
    return seqs


def find_junction(seq, hwt, twt, k=15):
    """Return (head_end, tail_start, source). head part = seq[:head_end], tail = seq[tail_start:]."""
    head_end = tail_start = None
    if hwt and len(seq) >= k:
        s = hwt.find(seq[:k])
        if s >= 0:
            n = 0
            while n < len(seq) and s + n < len(hwt) and seq[n] == hwt[s + n]:
                n += 1
            head_end = n
    if twt and len(seq) >= k:
        e = twt.rfind(seq[-k:])
        if e >= 0:
            n = 0  # extend backwards
            while n < len(seq) and e + k - 1 - n >= 0 and seq[-1 - n] == twt[e + k - 1 - n]:
                n += 1
            tail_start = len(seq) - n
    if head_end is not None and tail_start is not None:
        if tail_start < head_end:  # overlapping shared residues: split at the midpoint
            mid = (head_end + tail_start) // 2
            head_end, tail_start = mid, mid
        return head_end, tail_start, "both"
    if head_end is not None:
        return head_end, head_end, "head"
    if tail_start is not None:
        return tail_start, tail_start, "tail"
    return None, None, "none"


def pool(res, he, ts):
    L = len(res)
    out = {"mean": res.mean(0), "max": res.max(0)}
    if he is None or he <= 0 or ts >= L:
        out.update(head=out["mean"], tail=out["mean"], junction=out["mean"])
    else:
        out["head"] = res[:he].mean(0)
        out["tail"] = res[ts:].mean(0)
        out["junction"] = res[max(0, he - JW):min(L, ts + JW)].mean(0)
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("tag")
    ap.add_argument("--batch-tokens", type=int, default=12000)
    ap.add_argument("--junctions", default=None, help="CSV with seq_sha1, head_end, tail_start, src_mane_uniprot")
    a = ap.parse_args()
    out = P / "data/embeddings_pooled" / a.tag
    out.mkdir(parents=True, exist_ok=True)

    wt = pd.read_csv(P / "external/fusionpath/fusionpath/data/gene_wt_mane_sequences.csv")
    wt = dict(zip(wt.gene.str.upper(), wt.sequence.astype(str).str.replace("*", "", regex=False)))
    seq2genes = collect_sequences()
    seqs = sorted(seq2genes, key=len)
    junc = [find_junction(s, wt.get(seq2genes[s][0]), wt.get(seq2genes[s][1])) for s in seqs]
    if a.junctions:  # precomputed junctions (e.g. MANE + UniProt isoforms); MANE result kept if a sequence is missing
        jd = pd.read_csv(a.junctions)
        jmap = {h: (None if pd.isna(he) else int(he), None if pd.isna(ts) else int(ts), src)
                for h, he, ts, src in zip(jd.seq_sha1, jd.head_end, jd.tail_start, jd.src_mane_uniprot)}
        hs = [hashlib.sha1(s.encode()).hexdigest() for s in seqs]
        print(f'junctions from {a.junctions}: {sum(h in jmap for h in hs):,}/{len(hs):,} sequences', flush=True)
        junc = [jmap.get(h, j) for h, j in zip(hs, junc)]
    src = pd.Series([j[2] for j in junc]).value_counts().to_dict()
    print(f"{len(seqs):,} sequences; junction source: {src}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model_id)
    model = AutoModel.from_pretrained(a.model_id, dtype=torch.float16).to("cuda").eval()
    H = model.config.hidden_size
    pools = {p: np.zeros((len(seqs), H), np.float32) for p in POOLS}

    jobs = [(i, s, w) for i, seq in enumerate(seqs) for s, w in windows(seq)]
    remaining = {i: len(windows(seq)) for i, seq in enumerate(seqs)}
    acc = {}
    t0, k = time.time(), 0
    while k < len(jobs):
        batch, maxlen = [], 0
        while k < len(jobs):
            L = len(jobs[k][2]) + 2
            if batch and max(maxlen, L) * (len(batch) + 1) > a.batch_tokens:
                break
            batch.append(jobs[k]); maxlen = max(maxlen, L); k += 1
        enc = tok([w for _, _, w in batch], return_tensors="pt", padding=True).to("cuda")
        h = model(**enc).last_hidden_state.float()
        for b, (i, s, w) in enumerate(batch):
            if i not in acc:
                acc[i] = (np.zeros((len(seqs[i]), H), np.float32), np.zeros(len(seqs[i]), np.float32))
            acc[i][0][s:s + len(w)] += h[b, 1:len(w) + 1].cpu().numpy()
            acc[i][1][s:s + len(w)] += 1
            remaining[i] -= 1
            if remaining[i] == 0:
                ssum, cnt = acc.pop(i)
                he, ts, _ = junc[i]
                for p, v in pool(ssum / cnt[:, None], he, ts).items():
                    pools[p][i] = v
        if k % 4000 < len(batch):
            print(f"  {k:,}/{len(jobs):,} windows, {time.time() - t0:.0f}s", flush=True)

    assert not acc, "unfinished sequences"
    np.savez(out / "pools.npz", **{p: v.astype(np.float16) for p, v in pools.items()})
    pd.DataFrame({"seq_sha1": [hashlib.sha1(s.encode()).hexdigest() for s in seqs],
                  "length": [len(s) for s in seqs],
                  "head_end": [j[0] for j in junc], "tail_start": [j[1] for j in junc],
                  "junction_source": [j[2] for j in junc],
                  "n_windows": [len(windows(s)) for s in seqs]}).to_csv(out / "index.csv", index=False)
    print(f"done {len(seqs):,} sequences in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
