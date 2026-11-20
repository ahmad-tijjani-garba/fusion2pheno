"""Matched set: test-set sizes per assignment and the remaining provenance imbalance (no training)."""
from f2p_root import F2P_ROOT  # paths: see src/f2p_root.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd

P = F2P_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_v2 as T  # noqa: E402
import train_v2_checks as C  # noqa: E402

s, _ = T.load()
s = s.join(C.provenance(s, s), on=["pair", "sha1"])
pr = pd.read_csv(P / "results/round3/matched_predictions.csv.gz")
kept = set(pr.pair)
t = pr[pr.model == "shortcuts"].groupby("seed").agg(test_pairs=("pair", "size"), test_pos=("label", "sum"))
t["test_neg"] = t.test_pairs - t.test_pos
sizes = t.describe().loc[["min", "50%", "max"]]
print("test pairs per assignment (min/median/max):\n", sizes.to_string())
d = s[s.pair.isin(kept)]
g = d.groupby("pair").agg(label=("label", "first"), fuson_db=("fuson_db", "max"), fusiongdb2=("fusiongdb2", "max"),
                          fusionpath_db=("fusionpath_db", "max"), translated=("A_pipeline", "max"),
                          bp_singh2020=("bp_singh2020", "max"), nseq=("sha1", "nunique"))
imb = g.groupby("label")[["fuson_db", "fusiongdb2", "fusionpath_db", "translated", "bp_singh2020"]].mean().T
imb.columns = ["share_of_negative_pairs", "share_of_positive_pairs"]
imb.loc["median_sequences_per_pair"] = g.groupby("label").nseq.median().to_numpy()
print(imb.round(3).to_string())
out = pd.concat([sizes.T.assign(kind="test set size").reset_index().rename(columns={"index": "quantity"}),
                 imb.reset_index().rename(columns={"index": "quantity"}).assign(kind="matched set, by class")])
out.to_csv(P / "results/round3/matched_details.csv", index=False)
