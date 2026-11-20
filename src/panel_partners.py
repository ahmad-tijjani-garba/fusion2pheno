"""Round-3 review: panel calls under the gene-role-free labels by how many partner genes the training set contains. No training: rebuilds each calibration repeat's training set exactly as
score_panel_v2.py does (rng = default_rng(repeat); 15% of labelled pairs held out) and reads the saved calls.
Output: results/round3/panel_partners.csv and panel_partners.md
"""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd

P = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_v2 as T  # noqa: E402
import score_panel_v2 as SP  # noqa: E402

OUT = P / "results/round3"
OUT.mkdir(parents=True, exist_ok=True)
s, _ = T.load()
ps = SP.panel_isoforms()
fus = ps.groupby("fusion").agg(hgene=("hgene", "first"), tgene=("tgene", "first"))
sc = pd.read_csv(P / "results/v2_strict/panel/panel_scores.csv")
called = (sc.groupby(["model", "repeat", "fusion"]).called.mean() > 0.5).rename("called").reset_index()
pairs = np.array(sorted(s.pair.unique()))
rows = []
for r in sorted(sc.repeat.unique()):
    rng = np.random.default_rng(int(r))
    cal_pairs = set(rng.choice(pairs, size=int(0.15 * len(pairs)), replace=False))
    cal = s[s.pair.isin(cal_pairs)]
    tr = s[~s.pair.isin(cal_pairs) & ~s.sha1.isin(set(cal.sha1))]
    G = set(tr.hgene) | set(tr.tgene)
    seen = (fus.hgene.isin(G).astype(int) + fus.tgene.isin(G).astype(int)).rename("partners_seen")
    c = called[called.repeat == r].join(seen, on="fusion")
    rows.append(c)
c = pd.concat(rows)
tab = (c.groupby(["model", "partners_seen", "repeat"]).agg(n=("fusion", "size"), called=("called", "sum"))
       .groupby(["model", "partners_seen"]).mean().reset_index())
tab.to_csv(OUT / "panel_partners.csv", index=False)
keep = ["gene_name", "shortcuts", "esm2_mlp", "fuson_mlp", "esm2+shortcuts", "length"]
md = ["# Panel calls under gene-role-free labels by partner genes seen in training (mean over 5 calibration repeats)", "",
      tab[tab.model.isin(keep)].round(1).to_markdown(index=False), ""]
(OUT / "panel_partners.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
