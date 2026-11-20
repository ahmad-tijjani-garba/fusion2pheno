"""Frozen protein-language-model embeddings for every labelled sequence (inference only).

Mean-pooled last-layer embeddings. Sequences longer than CHUNK residues are split into
overlapping windows (stride CHUNK - OVERLAP); the per-residue embeddings of all windows
are averaged per residue position, then mean-pooled, so long fusions (e.g. KMT2A) keep
their 3' partner instead of being truncated.

Outputs (in data/embeddings/<model_tag>/):
  emb.npy       float16 [n_seqs, hidden]
  index.csv     seq_idx, seq_sha1, length, n_windows
Usage: python embed_sequences.py <hf_model_id> <model_tag> [--batch-tokens 12000]
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

P = F2P_ROOT
M = P / "data/fusionpath_rebuild/01_source_data/merged"
FILES = ["merged_fusions.filtered.hg38.formatted.labeled.csv",
         "merged_fusions.filtered.hg38.formatted.noncancerfix.labeled.csv"]
CHUNK, OVERLAP = 1022, 256


def labelled_sequences():
    seqs = set()
    for f in FILES:
        d = pd.read_csv(M / f, low_memory=False, usecols=["aa_seq", "cancer_label"])
        d = d[d.cancer_label.isin(["Benign", "Pathogenic"])]
        seqs |= set(d["aa_seq"].dropna().astype(str).str.replace("*", "", regex=False).str.upper())
    return sorted(seqs, key=len)


def windows(seq):
    if len(seq) <= CHUNK:
        return [(0, seq)]
    step = CHUNK - OVERLAP
    starts = list(range(0, len(seq) - CHUNK, step)) + [len(seq) - CHUNK]
    return [(s, seq[s:s + CHUNK]) for s in starts]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("tag")
    ap.add_argument("--batch-tokens", type=int, default=12000)
    ap.add_argument("--fasta", default=None, help="embed the sequences in this FASTA instead of the labelled FusionPath set")
    a = ap.parse_args()

    out = P / "data/embeddings" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model_id)
    model = AutoModel.from_pretrained(a.model_id, torch_dtype=torch.float16).to(dev).eval()
    hidden = model.config.hidden_size

    if a.fasta:
        seqs = sorted({l.strip().replace('*', '').upper() for l in open(a.fasta) if l.strip() and not l.startswith('>')}, key=len)
    else:
        seqs = labelled_sequences()
    print(f"{len(seqs):,} unique labelled sequences; GPU {torch.cuda.get_device_name(0)}", flush=True)

    # flatten into windows: (seq_idx, start, window_seq)
    jobs = [(i, s, w) for i, seq in enumerate(seqs) for s, w in windows(seq)]
    pooled = np.zeros((len(seqs), hidden), np.float32)
    # accumulate per-residue sums and counts only for multi-window sequences
    multi = {i for i, seq in enumerate(seqs) if len(seq) > CHUNK}
    acc = {i: (np.zeros((len(seqs[i]), hidden), np.float32), np.zeros(len(seqs[i]), np.float32)) for i in multi}

    t0, k = time.time(), 0
    while k < len(jobs):
        batch, maxlen = [], 0
        while k < len(jobs):
            L = len(jobs[k][2]) + 2
            if batch and max(maxlen, L) * (len(batch) + 1) > a.batch_tokens:
                break
            batch.append(jobs[k]); maxlen = max(maxlen, L); k += 1
        enc = tok([w for _, _, w in batch], return_tensors="pt", padding=True).to(dev)
        h = model(**enc).last_hidden_state.float()  # [B, T, H]
        for b, (i, s, w) in enumerate(batch):
            res = h[b, 1:len(w) + 1].cpu().numpy()  # drop CLS/EOS
            if i in multi:
                acc[i][0][s:s + len(w)] += res
                acc[i][1][s:s + len(w)] += 1
            else:
                pooled[i] = res.mean(0)
        if k % 2000 < len(batch):
            print(f"  {k:,}/{len(jobs):,} windows, {time.time() - t0:.0f}s", flush=True)

    for i, (s, c) in acc.items():
        pooled[i] = (s / c[:, None]).mean(0)

    np.save(out / "emb.npy", pooled.astype(np.float16))
    pd.DataFrame({"seq_idx": range(len(seqs)),
                  "seq_sha1": [hashlib.sha1(s.encode()).hexdigest() for s in seqs],
                  "length": [len(s) for s in seqs],
                  "n_windows": [len(windows(s)) for s in seqs]}).to_csv(out / "index.csv", index=False)
    print(f"done: {len(seqs):,} sequences, {len(multi):,} chunked, {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
