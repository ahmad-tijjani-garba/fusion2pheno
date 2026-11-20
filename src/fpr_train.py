"""FusionPath rebuild (triple-branch network) and domain-retention baseline on the FusionPath-style splits.

FusionPath's released code (commit 6f0733d) lacks its network class, weights and vocabularies, so the architecture is
rebuilt from the repository notebook (sections 6-7): a sequence branch (FusOn-pLM fusion + ProtBERT wild-type head and
tail embeddings) and two attention branches over domain and GO tokens, concatenated into a classifier. Domains are Pfam
calls (not full InterProScan); GO terms come from pfam2go. Token vocabularies are built from each training split only.
Variants: full (sequence + domains + GO), seq_dom (no GO), dom_go (domains + GO, no embeddings).
Domain-retention baseline: logistic regression on 11 Pfam retention features (no gene identity, no length).
Self-contained (numpy, pandas, torch, scikit-learn); inputs from export_domain_features.py.
Usage: python fpr_train.py <data dir> <out dir> [--splits name ...]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAXTOK, EMB, HID, DROP, LR, WD, BATCH, EPOCHS, PATIENCE = 50, 64, 256, 0.4, 1e-4, 1e-4, 64, 25, 5


class FusionNetTriple(nn.Module):
    def __init__(self, seq_dim, vocab_dom, vocab_go, use_seq=True, use_dom=True, use_go=True):
        super().__init__()
        self.use = (use_seq, use_dom, use_go)
        dim = 0
        if use_seq:
            self.seq_encoder = nn.Sequential(nn.Linear(seq_dim, HID), nn.ReLU(), nn.Dropout(DROP)); dim += HID
        if use_dom:
            self.dom_emb = nn.Embedding(vocab_dom, EMB, padding_idx=0)
            self.dom_attn = nn.Sequential(nn.Linear(EMB, 1), nn.Tanh()); dim += EMB
        if use_go:
            self.go_emb = nn.Embedding(vocab_go, EMB, padding_idx=0)
            self.go_attn = nn.Sequential(nn.Linear(EMB, 1), nn.Tanh()); dim += EMB
        self.classifier = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(DROP), nn.Linear(256, 1))

    @staticmethod
    def context(emb, attn, ids):
        e = emb(ids)                                   # [B, L, EMB]
        a = attn(e).squeeze(-1)                        # [B, L]
        a = a.masked_fill(ids == 0, -1e9)
        w = torch.softmax(a, dim=1) * (ids != 0)       # rows with no token -> all-zero weights
        return (w.unsqueeze(-1) * e).sum(1)

    def forward(self, seq, dom, go):
        f = []
        if self.use[0]:
            f.append(self.seq_encoder(seq))
        if self.use[1]:
            f.append(self.context(self.dom_emb, self.dom_attn, dom))
        if self.use[2]:
            f.append(self.context(self.go_emb, self.go_attn, go))
        return self.classifier(torch.cat(f, 1)).squeeze(1)


def tokens(strings, vocab):
    out = np.zeros((len(strings), MAXTOK), np.int64)
    for i, s in enumerate(strings):
        ids = [vocab.get(t, 1) for t in s.split()][:MAXTOK]
        out[i, :len(ids)] = ids
    return out


def vocab_of(strings):
    v = {"<PAD>": 0, "<UNK>": 1}
    for s in strings:
        for t in s.split():
            v.setdefault(t, len(v))
    return v


def fit_net(Xtr, Dtr, Gtr, ytr, Xva, Dva, Gva, yva, Xte, Dte, Gte, vd, vg, use, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = FusionNetTriple(Xtr.shape[1], vd, vg, *use).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    pw = torch.tensor([(1 - ytr.mean()) / max(ytr.mean(), 1e-6)], device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    T = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=DEV)  # noqa: E731
    Xtr_, Dtr_, Gtr_, ytr_ = T(Xtr), T(Dtr, torch.long), T(Gtr, torch.long), T(ytr)
    ev = lambda X, Dm, G: torch.sigmoid(torch.cat([net(T(X[i:i + 4096]), T(Dm[i:i + 4096], torch.long), T(G[i:i + 4096], torch.long))  # noqa: E731
                                                  for i in range(0, len(X), 4096)])).cpu().numpy()
    g = torch.Generator().manual_seed(seed)
    best, state, bad = -1, None, 0
    for _ in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(ytr), generator=g).to(DEV)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            lossf(net(Xtr_[b], Dtr_[b], Gtr_[b]), ytr_[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pva = ev(Xva, Dva, Gva)
        score = average_precision_score(yva, pva) if len(np.unique(yva)) > 1 else 0
        if score > best:
            best, bad = score, 0
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        return ev(Xte, Dte, Gte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    a = ap.parse_args()
    D, OUT = Path(a.data), Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    B = np.load(D / "fpr_bundle.npz", allow_pickle=True)
    meta = pd.read_csv(D / "fpr_meta.csv.gz")
    S = json.load(open(D / "fpr_splits.json"))
    pos = {r: i for i, r in enumerate(B["row_id"])}
    Z, y, dom, go, R = B["Z"], B["label"], B["dom"], B["go"], B["ret"]
    print(f"device {DEV}; {len(y):,} records; Z {Z.shape}; splits {len(S)}", flush=True)
    t0 = time.time()
    preds, rows = [], []
    for name, (tr, va, te) in S.items():
        if a.splits and name not in a.splits:
            continue
        itr, iva, ite = ([pos[r] for r in x] for x in (tr, va, te))
        ytr, yva, yte = y[itr], y[iva], y[ite]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        sc = StandardScaler().fit(Z[itr].astype(np.float32))
        Xtr, Xva, Xte = (sc.transform(Z[i].astype(np.float32)).astype(np.float32) for i in (itr, iva, ite))
        vd, vg = vocab_of(dom[itr]), vocab_of(go[itr])
        Dtr, Dva, Dte = (tokens(dom[i], vd) for i in (itr, iva, ite))
        Gtr, Gva, Gte = (tokens(go[i], vg) for i in (itr, iva, ite))
        out = {}
        for vname, use in [("fusionpath_rebuild/full", (True, True, True)), ("fusionpath_rebuild/seq_dom", (True, True, False)),
                           ("fusionpath_rebuild/dom_go", (False, True, True))]:
            out[vname] = np.mean([fit_net(Xtr, Dtr, Gtr, ytr, Xva, Dva, Gva, yva, Xte, Dte, Gte, len(vd), len(vg), use, s)
                                  for s in a.seeds], axis=0)
        rs = StandardScaler().fit(R[itr])
        out["domain_retention"] = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0).fit(
            rs.transform(R[itr]), ytr).predict_proba(rs.transform(R[ite]))[:, 1]
        m = meta.iloc[ite]
        tg = set(meta.iloc[itr].hgene) | set(meta.iloc[itr].tgene)
        seen = m.hgene.isin(tg).astype(int).to_numpy() + m.tgene.isin(tg).astype(int).to_numpy()
        for model, p in out.items():
            rows.append({"labelset": "as_published", "split": name, "model": model, "n_test": len(yte), "pos": int(yte.sum()),
                         "AUROC": roc_auc_score(yte, p), "AUPRC": average_precision_score(yte, p)})
            preds.append(pd.DataFrame({"labelset": "as_published", "split": name, "model": model, "row_id": m.row_id.to_numpy(),
                                       "hgene": m.hgene.to_numpy(), "tgene": m.tgene.to_numpy(), "pair": m.pair.to_numpy(),
                                       "label": yte, "score": p, "genes_seen": seen}))
        print(f"{name}: " + ", ".join(f"{k} {roc_auc_score(yte, v):.3f}" for k, v in out.items()) +
              f" | vocab dom {len(vd)}, go {len(vg)} | {time.time() - t0:.0f}s", flush=True)
        pd.concat(preds).to_csv(OUT / "predictions_FPR.csv.gz", index=False)
        pd.DataFrame(rows).to_csv(OUT / "metrics_FPR.csv", index=False)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
