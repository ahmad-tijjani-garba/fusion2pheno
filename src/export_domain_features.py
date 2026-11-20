"""Build the inputs for (1) the FusionPath rebuild and (2) the domain-retention baseline, from the Pfam calls.

Runs on CPE (no training). Writes data/domains/:
  fpr_bundle.npz       Z (FusOn-pLM native fusion embedding | ProtBERT wild-type head | tail | missing flags, float16),
                       dom/go token strings per record, retention features, labels  (FusionPath-style labels, as published)
  fpr_meta.csv.gz      row_id, hgene, tgene, pair, label (same rows and order as Z)
  fpr_splits.json      the 13 FusionPath-style splits (row_id lists), identical to fp_benchmark_rev.py
  v2_domain_features.csv.gz  retention features per (seq_sha1, hgene, tgene) for the gene-role-free sequences
Domain tokens: Pfam accessions found on the fusion protein (gathering thresholds), in sequence order, max 50.
GO tokens: GO terms mapped from those Pfam families by pfam2go (InterPro2GO-derived), max 50.
Retention: for each partner, Pfam domains of its MANE wild-type protein and how many of them the fusion keeps.
"""
from f2p_root import F2P_CACHE, F2P_ROOT  # paths: see src/f2p_root.py
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

P = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fp_benchmark_rev as FB  # noqa: E402

D = P / "data/domains"
PF = F2P_CACHE / "pfam"
KINASE = {"PF00069", "PF07714"}
RET_COLS = ["h_wt_domains", "h_retained", "h_retained_frac", "h_wt_missing", "t_wt_domains", "t_retained",
            "t_retained_frac", "t_wt_missing", "fusion_domains", "fusion_novel_domains", "kinase_domain_in_fusion"]

hits = pd.read_csv(D / "pfam_hits.csv.gz").sort_values(["seq_sha1", "env_from"])
doms = hits.groupby("seq_sha1").pfam_acc.apply(list).to_dict()
p2go = {}
for line in open(PF / "pfam2go"):
    if line.startswith("Pfam:"):
        acc = line.split()[0].split(":")[1]
        go = line.strip().rsplit(";", 1)[-1].strip()
        p2go.setdefault(acc, []).append(go)
wt = FB.wt_lookup()  # gene symbol (raw and HGNC-normalised) -> sha1 of the MANE wild-type protein


def retention(fsha, genes_h, genes_t):
    f = Counter(doms.get(fsha, []))
    out = []
    sides = []
    for gs in (genes_h, genes_t):
        w = next((wt[g] for g in gs if g in wt), None)
        c = Counter(doms.get(w, [])) if w else Counter()
        ret = sum(min(n, f[a]) for a, n in c.items())
        n = sum(c.values())
        out += [n, ret, ret / n if n else 0.0, 0 if w else 1]
        sides.append(c)
    novel = sum(n for a, n in f.items() if a not in sides[0] and a not in sides[1])
    out += [sum(f.values()), novel, int(bool(KINASE & set(f)))]
    return out


def go_tokens(acc_list):
    seen, out = set(), []
    for a in acc_list:
        for g in p2go.get(a, []):
            if g not in seen:
                seen.add(g); out.append(g)
    return out[:50]


# ---------------------------------------------------------------- FusionPath-style (as published), same rows as the benchmark
tags = ["esm2_650m", "fuson_plm", "fuson_plm_native", "esm2_3b", "prot_t5"]
emb = {t: FB.load_emb(t) for t in tags}
W, wlut = FB.load_emb("protbert_wt")
d = FB.load("as_published")
ok = np.logical_and.reduce([d.sha1.isin(emb[t][1]).to_numpy() for t in tags])
d = d[ok]
S = FB.splits(d, 10)
Ef, lutf = emb["fuson_plm_native"]
Z = FB.fusionpath_like(d, Ef, lutf, W, wlut, wt).astype(np.float16)
dom_tok = [" ".join(doms.get(s, [])[:50]) for s in d.sha1]
go_tok = [" ".join(go_tokens(doms.get(s, []))) for s in d.sha1]
R = np.array([retention(s, (rh, h), (rt, t)) for s, rh, h, rt, t in zip(d.sha1, d.raw_h, d.hgene, d.raw_t, d.tgene)],
             dtype=np.float32)
np.savez_compressed(D / "fpr_bundle.npz", Z=Z, label=d.label.to_numpy(), row_id=d.index.to_numpy(),
                    dom=np.array(dom_tok, dtype=object), go=np.array(go_tok, dtype=object), ret=R,
                    ret_cols=np.array(RET_COLS))
d[["hgene", "tgene", "pair", "label"]].reset_index().to_csv(D / "fpr_meta.csv.gz", index=False)
json.dump({k: [list(map(int, x)) for x in v] for k, v in S.items()}, open(D / "fpr_splits.json", "w"))
print(f"FusionPath-style: {len(d):,} records, Z {Z.shape}, {np.mean([len(x) > 0 for x in dom_tok]):.1%} with a Pfam domain, "
      f"{np.mean([len(x) > 0 for x in go_tok]):.1%} with a GO term", flush=True)
print(pd.DataFrame(R, columns=RET_COLS).describe().round(2).T.to_string())

# ---------------------------------------------------------------- gene-role-free sequences
import train_v2 as T  # noqa: E402
s, _ = T.load()
u = s.drop_duplicates(["sha1", "hgene", "tgene"])[["sha1", "hgene", "tgene"]]
V = pd.DataFrame([retention(sh, (h,), (t,)) for sh, h, t in zip(u.sha1, u.hgene, u.tgene)], columns=RET_COLS)
V.insert(0, "seq_sha1", u.sha1.to_numpy()); V.insert(1, "hgene", u.hgene.to_numpy()); V.insert(2, "tgene", u.tgene.to_numpy())
V.to_csv(D / "v2_domain_features.csv.gz", index=False)
print(f"gene-role-free: {len(V):,} sequence/pair rows; {(V.fusion_domains > 0).mean():.1%} with a Pfam domain")
