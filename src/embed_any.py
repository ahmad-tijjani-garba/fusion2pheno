"""Revision: frozen mean-pooled embeddings for any supported pLM (inference only).

Model families: esm (ESM-2, FusOn-pLM; AutoModel), t5 (ProtT5 encoder; residues space-separated, U/Z/O/B -> X),
bert (ProtBERT; residues space-separated, U/Z/O/B -> X). Last hidden layer; special tokens excluded from the mean.
Sequences longer than --chunk residues are split into windows overlapping by --overlap residues; per-residue
embeddings are averaged over windows, then mean-pooled (same scheme as embed_sequences.py).

Inputs : --labelled (the labelled FusionPath sequences) and/or one or more --fasta files; the union is embedded.
Outputs: data/embeddings/<tag>/{emb.npy (float16), index.csv (seq_idx, seq_sha1, length, n_windows)}
Usage  : python embed_any.py <hf_model_id> <tag> --family esm|t5|bert [--chunk 1022] [--labelled] [--fasta f ...]
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import argparse
import hashlib
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float16 if DEV == "cuda" else torch.float32  # half precision only on GPU

P = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_sequences import labelled_sequences  # noqa: E402


def windows(seq, chunk, overlap):
    if len(seq) <= chunk:
        return [(0, seq)]
    step = chunk - overlap
    starts = list(range(0, len(seq) - chunk, step)) + [len(seq) - chunk]
    return [(s, seq[s:s + chunk]) for s in starts]


def load(model_id, family):
    from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer, T5EncoderModel
    if family == "esm":
        return AutoTokenizer.from_pretrained(model_id), AutoModel.from_pretrained(model_id, dtype=DT)
    if family == "t5":
        # ProtT5's SentencePiece vocabulary is one token per residue; read it from spiece.model directly so the
        # sentencepiece package is not needed (ids: <pad> 0, </s> 1, <unk> 2, then "▁A", "▁L", ...)
        from huggingface_hub import hf_hub_download
        b = open(hf_hub_download(model_id, "spiece.model", local_files_only=True), "rb").read()
        pieces, i = [], 0
        while i < len(b) and b[i] == 0x0A:
            i += 1
            ln, shift = 0, 0
            while True:
                c = b[i]; i += 1; ln |= (c & 0x7F) << shift; shift += 7
                if c < 0x80:
                    break
            msg = b[i:i + ln]; i += ln
            if msg[:1] == b"\x0a":
                pieces.append(msg[2:2 + msg[1]].decode("utf-8"))
        vocab = {p.lstrip("▁"): k for k, p in enumerate(pieces) if p.startswith("▁")}
        assert pieces[:3] == ["<pad>", "</s>", "<unk>"] and len(vocab) == 25, pieces
        return vocab, T5EncoderModel.from_pretrained(model_id, dtype=DT)
    return BertTokenizer.from_pretrained(model_id, do_lower_case=False), BertModel.from_pretrained(model_id, dtype=DT)


def prep(w, family):
    return w if family == "esm" else " ".join(re.sub(r"[UZOB]", "X", w))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("tag")
    ap.add_argument("--family", choices=["esm", "t5", "bert"], required=True)
    ap.add_argument("--chunk", type=int, default=1022)
    ap.add_argument("--overlap", type=int, default=256)
    ap.add_argument("--batch-tokens", type=int, default=12000)
    ap.add_argument("--labelled", action="store_true")
    ap.add_argument("--fasta", nargs="*", default=[])
    a = ap.parse_args()

    seqs = set(labelled_sequences()) if a.labelled else set()
    for f in a.fasta:
        seqs |= {l.strip().replace("*", "").upper() for l in open(f) if l.strip() and not l.startswith(">")}
    seqs = sorted(seqs, key=len)
    out = P / "data/embeddings" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    tok, model = load(a.model_id, a.family)
    model = model.to(DEV).eval()
    torch.set_num_threads(int(__import__("os").environ.get("SLURM_CPUS_PER_TASK", 8)))
    hidden = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.d_model
    print(f"{a.tag}: {len(seqs):,} sequences, chunk {a.chunk}, device {torch.cuda.get_device_name(0) if DEV == 'cuda' else 'cpu'}", flush=True)

    jobs = [(i, s, w) for i, seq in enumerate(seqs) for s, w in windows(seq, a.chunk, a.overlap)]
    pooled = np.zeros((len(seqs), hidden), np.float32)
    multi = {i for i, seq in enumerate(seqs) if len(seq) > a.chunk}
    left = {i: len(windows(seqs[i], a.chunk, a.overlap)) for i in multi}
    acc = {}  # per-residue buffers, created on a sequence's first window and freed after its last
    n_multi = len(multi)
    t0, k = time.time(), 0
    while k < len(jobs):
        batch, maxlen = [], 0
        while k < len(jobs):
            L = len(jobs[k][2]) + 2
            if batch and max(maxlen, L) * (len(batch) + 1) > a.batch_tokens:
                break
            batch.append(jobs[k]); maxlen = max(maxlen, L); k += 1
        if a.family == "t5":  # residue tokens + </s>, padded with <pad> (0)
            ids = [[tok.get(c, 2) for c in re.sub(r"[UZOB]", "X", w)] + [1] for _, _, w in batch]
            L = max(map(len, ids))
            ii = torch.tensor([x + [0] * (L - len(x)) for x in ids], device=DEV)
            enc = {"input_ids": ii, "attention_mask": (torch.arange(L, device=DEV)[None, :]
                                                        < torch.tensor([len(x) for x in ids], device=DEV)[:, None]).long()}
        else:
            enc = tok([prep(w, a.family) for _, _, w in batch], return_tensors="pt", padding=True,
                      add_special_tokens=True).to(DEV)
        h = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state.float()
        off = 0 if a.family == "t5" else 1  # ESM and BERT start with a CLS token; T5 ends with </s> only
        for b, (i, s, w) in enumerate(batch):
            res = h[b, off:off + len(w)].cpu().numpy()
            if i in multi:
                if i not in acc:
                    acc[i] = (np.zeros((len(seqs[i]), hidden), np.float32), np.zeros(len(seqs[i]), np.float32))
                acc[i][0][s:s + len(w)] += res
                acc[i][1][s:s + len(w)] += 1
                left[i] -= 1
                if left[i] == 0:
                    sm, c = acc.pop(i)
                    pooled[i] = (sm / c[:, None]).mean(0)
            else:
                pooled[i] = res.mean(0)
        if k % 4000 < len(batch):
            print(f"  {k:,}/{len(jobs):,} windows, {time.time() - t0:.0f}s", flush=True)
    assert not acc, "unfinished windowed sequences"
    multi = range(n_multi)
    assert np.isfinite(pooled).all()
    np.save(out / "emb.npy", pooled.astype(np.float16))
    pd.DataFrame({"seq_idx": range(len(seqs)), "seq_sha1": [hashlib.sha1(s.encode()).hexdigest() for s in seqs],
                  "length": [len(s) for s in seqs],
                  "n_windows": [len(windows(s, a.chunk, a.overlap)) for s in seqs]}).to_csv(out / "index.csv", index=False)
    print(f"done: {len(seqs):,} sequences, {len(multi):,} windowed, {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
