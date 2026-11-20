"""Revision: stronger probes of ESM-2 650M on the gene-role-free labels.

attn : light-attention head (Staerk et al. style) over frozen per-residue embeddings (1,022-residue windows, averaged
       per residue), instead of mean pooling. Conv1d(1280 -> 2x256, k=9) gives values and attention logits; softmax
       over residues -> weighted sum, concatenated with max-pooled values -> MLP -> logit.
lora : LoRA fine-tune of ESM-2 650M (rank 8, alpha 16, on query and value projections of every layer) with a
       linear head on the mean-pooled final layer. Training uses a random <=1,022-residue window per sequence per
       epoch; prediction averages over all windows.
Same pairs, gene assignments, pair weights and pair-level scoring as train_v2.py; early stopping on validation pair AUROC.
Outputs: results/v2_rev/probes/{predictions.csv.gz, summary.md}
"""
from f2p_root import F2P_CACHE, F2P_ROOT  # paths: see src/f2p_root.py
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

P = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_v2 as T  # noqa: E402
import train_v2_checks as C  # noqa: E402
import train_v2_matched as MT  # noqa: E402
from embed_sequences import labelled_sequences  # noqa: E402

OUT = P / "results/v2_rev/probes"
# ~6.5 GB per-residue cache: kept on the shared server, outside the home-disk quota
RES = F2P_CACHE / "esm2_650m_residues"
MODEL = "facebook/esm2_t33_650M_UR50D"
DEV = "cuda"
CHUNK, OVERLAP = 1022, 256


def windows(seq):
    if len(seq) <= CHUNK:
        return [(0, seq)]
    step = CHUNK - OVERLAP
    starts = list(range(0, len(seq) - CHUNK, step)) + [len(seq) - CHUNK]
    return [(s, seq[s:s + CHUNK]) for s in starts]


def sequences(s):
    import hashlib
    seqs = {hashlib.sha1(x.encode()).hexdigest(): x for x in labelled_sequences()}
    for line in open(P / "data/labels_v2/strict/new_sequences.fasta"):
        if line.strip() and not line.startswith(">"):
            x = line.strip().replace("*", "").upper()
            seqs[hashlib.sha1(x.encode()).hexdigest()] = x
    return {sh: seqs[sh] for sh in s.sha1.unique()}


@torch.no_grad()
def residue_embeddings(seqs):
    """Per-residue ESM-2 650M embeddings (float16 memmap), cached; returns (memmap, offsets)."""
    from transformers import AutoModel, AutoTokenizer
    ix_f = RES / "index.csv"
    if ix_f.exists():
        ix = pd.read_csv(ix_f)
        if set(seqs) <= set(ix.sha1):
            mm = np.memmap(RES / "res.f16", dtype=np.float16, mode="r", shape=(int(ix.end.max()), 1280))
            return mm, {r.sha1: (r.start, r.end) for r in ix.itertuples()}
    RES.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).to(DEV).eval()
    order = sorted(seqs, key=lambda k: len(seqs[k]))
    starts = np.cumsum([0] + [len(seqs[k]) for k in order])
    mm = np.memmap(RES / "res.f16", dtype=np.float16, mode="w+", shape=(int(starts[-1]), 1280))
    t0 = time.time()
    for n, k in enumerate(order):
        seq = seqs[k]
        acc = np.zeros((len(seq), 1280), np.float32)
        cnt = np.zeros(len(seq), np.float32)
        for st, w in windows(seq):
            enc = tok([w], return_tensors="pt").to(DEV)
            h = model(**enc).last_hidden_state[0, 1:len(w) + 1].float().cpu().numpy()
            acc[st:st + len(w)] += h
            cnt[st:st + len(w)] += 1
        mm[starts[n]:starts[n + 1]] = (acc / cnt[:, None]).astype(np.float16)
        if n % 500 == 0:
            print(f"  residues {n}/{len(order)} {time.time() - t0:.0f}s", flush=True)
    mm.flush()
    pd.DataFrame({"sha1": order, "start": starts[:-1], "end": starts[1:]}).to_csv(ix_f, index=False)
    del model
    torch.cuda.empty_cache()
    return residue_embeddings(seqs)


class LightAttention(nn.Module):
    def __init__(self, d=1280, h=256, k=9):
        super().__init__()
        self.val = nn.Conv1d(d, h, k, padding=k // 2)
        self.att = nn.Conv1d(d, h, k, padding=k // 2)
        self.drop = nn.Dropout(0.25)
        self.out = nn.Sequential(nn.Linear(2 * h, 64), nn.ReLU(), nn.Dropout(0.25), nn.Linear(64, 1))

    def forward(self, x, mask):  # x [B, L, D], mask [B, L] (True = residue)
        x = self.drop(x.transpose(1, 2))
        v, a = self.val(x), self.att(x)
        a = a.masked_fill(~mask[:, None, :], -1e4)
        o1 = (torch.softmax(a, -1) * v).sum(-1)
        o2 = v.masked_fill(~mask[:, None, :], -1e4).max(-1).values
        return self.out(torch.cat([o1, o2], 1)).squeeze(1)


def batches(keys, mm, offs, size=16):
    for i in range(0, len(keys), size):
        ks = keys[i:i + size]
        L = max(offs[k][1] - offs[k][0] for k in ks)
        x = np.zeros((len(ks), L, 1280), np.float32)
        m = np.zeros((len(ks), L), bool)
        for j, k in enumerate(ks):
            a, b = offs[k]
            x[j, :b - a] = mm[a:b]
            m[j, :b - a] = True
        yield ks, torch.tensor(x, device=DEV), torch.tensor(m, device=DEV)


def fit_attention(tr, va, te, mm, offs, seed):
    torch.manual_seed(seed)
    net = LightAttention().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-2)
    w = torch.tensor(T.weights(tr), dtype=torch.float32, device=DEV)
    y = torch.tensor(tr.label.to_numpy(), dtype=torch.float32, device=DEV)
    keys = tr.sha1.tolist()
    order = np.argsort([offs[k][1] - offs[k][0] for k in keys])  # length-sorted batches, shuffled batch order

    def predict(df):
        net.eval()
        uk = df.sha1.unique().tolist()
        uk.sort(key=lambda k: offs[k][1] - offs[k][0])
        sc = {}
        with torch.no_grad():
            for ks, x, m in batches(uk, mm, offs):
                for k, p in zip(ks, torch.sigmoid(net(x, m)).cpu().numpy()):
                    sc[k] = p
        return df.sha1.map(sc).to_numpy()

    best, state, patience = -1, None, 0
    rng = np.random.default_rng(seed)
    for epoch in range(40):
        net.train()
        chunks = [order[i:i + 16] for i in range(0, len(order), 16)]
        rng.shuffle(chunks)
        for idx in chunks:
            ks = [keys[i] for i in idx]
            _, x, m = next(batches(ks, mm, offs, size=len(ks)))
            loss = F.binary_cross_entropy_with_logits(net(x, m), y[idx], reduction="none")
            opt.zero_grad()
            ((loss * w[idx]).sum() / w[idx].sum()).backward()
            opt.step()
        a = T.pair_auc(va, predict(va))
        if a > best:
            best, patience = a, 0
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 6:
                break
    net.load_state_dict(state)
    return predict(te)


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def fit_lora(tr, va, te, seqs, seed, epochs=4):
    from transformers import AutoModel, AutoTokenizer
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float32)
    for p in model.parameters():
        p.requires_grad = False
    for layer in model.encoder.layer:
        att = layer.attention.self
        att.query, att.value = LoRALinear(att.query), LoRALinear(att.value)
    model = model.to(DEV)
    head = nn.Linear(1280, 1).to(DEV)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": params, "lr": 2e-4}, {"params": head.parameters(), "lr": 1e-3}], weight_decay=1e-2)
    w_all = T.weights(tr)
    rng = np.random.default_rng(seed)

    def embed(batch_seqs, train):
        enc = tok(batch_seqs, return_tensors="pt", padding=True).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(**enc).last_hidden_state
        m = enc["attention_mask"].clone()
        m[:, 0] = 0
        m[torch.arange(len(batch_seqs)), enc["attention_mask"].sum(1) - 1] = 0  # drop CLS and EOS
        return (h.float() * m[..., None]).sum(1) / m.sum(1, keepdim=True)

    def predict(df):
        model.eval()
        sc = {}
        with torch.no_grad():
            for k in df.sha1.unique():
                ws = [w for _, w in windows(seqs[k])]
                lens = torch.tensor([len(w) for w in ws], dtype=torch.float32, device=DEV)
                e = torch.cat([embed(ws[i:i + 4], False) for i in range(0, len(ws), 4)])
                sc[k] = torch.sigmoid(head((e * lens[:, None]).sum(0) / lens.sum())).item()
        return df.sha1.map(sc).to_numpy()

    best, state, patience = -1, None, 0
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(len(tr))
        for i in range(0, len(perm), 4):
            idx = perm[i:i + 4]
            crops = []
            for j in idx:
                sq = seqs[tr.sha1.iloc[j]]
                st = rng.integers(0, max(1, len(sq) - CHUNK + 1))
                crops.append(sq[st:st + CHUNK])
            logit = head(embed(crops, True)).squeeze(1)
            y = torch.tensor(tr.label.to_numpy()[idx], dtype=torch.float32, device=DEV)
            w = torch.tensor(w_all[idx], dtype=torch.float32, device=DEV)
            loss = (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * w).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        a = T.pair_auc(va, predict(va))
        print(f"    lora epoch {epoch}: val pair AUROC {a:.3f}", flush=True)
        if a > best:
            best, patience = a, 0
            state = ({k: v.detach().clone() for k, v in model.state_dict().items() if "lora" in k or ".A" in k or ".B" in k},
                     {k: v.detach().clone() for k, v in head.state_dict().items()})
        else:
            patience += 1
            if patience >= 2:
                break
    model.load_state_dict(state[0], strict=False)
    head.load_state_dict(state[1])
    out = predict(te)
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", nargs="*", default=["attn", "lora"])
    ap.add_argument("--pooled-subsets", nargs="*", default=["AB_all", "AB_nonRT"])
    ap.add_argument("--matched-subsets", nargs="*", default=["A_fusionpath", "fusiongdb2", "fusionpath_db", "fuson_db"])
    ap.add_argument("--attn-seeds", type=int, nargs="*", default=T.SEEDS)
    ap.add_argument("--lora-seeds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    ap.add_argument("--lora-subsets", nargs="*", default=["AB_all", "A_fusionpath"])
    ap.add_argument("--run-name", default="all")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    s, _ = T.load()
    s = s.join(C.provenance(s, s), on=["pair", "sha1"])
    seqs = sequences(s)
    mm, offs = residue_embeddings(seqs) if "attn" in a.probes else (None, None)
    allpairs = s.drop_duplicates("pair")[["pair", "hgene", "tgene"]]
    preds = []
    plan = [(sub, "attn", seed) for sub in a.pooled_subsets + a.matched_subsets for seed in a.attn_seeds if "attn" in a.probes]
    plan += [(sub, "lora", seed) for sub in a.lora_subsets for seed in a.lora_seeds if "lora" in a.probes]
    for sub, probe, seed in plan:
        if sub in a.matched_subsets:
            d0 = MT.subset(s, sub)
        else:
            d0 = s if sub == "AB_all" else s[s.rt == 0] if sub == "AB_nonRT" else s[s.set == "A_pipeline"]
        part, _ = T.split(allpairs, seed)
        d = d0.assign(part=d0.pair.map(part))
        test_seqs = set(d.sha1[d.part == "test"])
        tr = d[(d.part == "train") & ~d.sha1.isin(test_seqs)]
        va = d[(d.part == "val") & ~d.sha1.isin(test_seqs)]
        te = d[d.part == "test"]
        if min(tr.label.nunique(), va.label.nunique(), te.label.nunique()) < 2:
            continue
        p = fit_attention(tr, va, te, mm, offs, seed) if probe == "attn" else fit_lora(tr, va, te, seqs, seed)
        preds.append(T.pair_scores(te, p).assign(subset=sub, seed=seed, model=f"esm2_{probe}", C=np.nan).reset_index())
        print(f"{probe} {sub} seed {seed}: test {te.pair.nunique()} pairs, pair AUROC {T.pair_auc(te, p):.3f}, "
              f"{time.time() - t0:.0f}s", flush=True)
        pd.concat(preds).to_csv(OUT / f"predictions_{a.run_name}.csv.gz", index=False)
    pr = T.mark_both_unseen(pd.concat(preds), s)
    pr.to_csv(OUT / f"predictions_{a.run_name}.csv.gz", index=False)
    ps = T.per_seed(pr)
    ps.to_csv(OUT / f"metrics_per_seed_{a.run_name}.csv", index=False)
    (OUT / f"summary_{a.run_name}.md").write_text("# Stronger probes of ESM-2 650M (revision)\n\n"
                                    + ps.groupby(["subset", "model"]).auroc.agg(["mean", "std", "count"]).round(3).to_markdown() + "\n")
    print((OUT / f"summary_{a.run_name}.md").read_text())
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
