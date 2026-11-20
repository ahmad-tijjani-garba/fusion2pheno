"""Figures 1-5 for the Fusion2Pheno manuscript, drawn from the result tables in manuscript/results/.

Usage: python make_figures.py <results dir> <figures out dir> <plotted-data out dir>
Each figure is saved as an RGB PNG at 600 dpi (FigureN.png); its exact plotted values go to
<plotted-data out dir>/FigureN_data.csv.

Fig 1: study design (schematic), placed in Methods, Study design.
Fig 2: AUROC by split with gene-resampled CIs (results/rev1/reanalysis/fp.csv).
Fig 3: held-out genes on the FusionPath-style labels, every model of Table 3 (reanalysis/fp.csv and fpr.csv).
Fig 4: gene-role-free labels: (A) AUROC per training set; (B) gain from adding each embedding to the shortcut features
       (results/rev1/reanalysis/v2.csv).
Fig 5: paediatric panel calls under both label sets (results/rev1/fp_rev/panel, results/rev1/v2_strict/panel).

Style: one encoding per model across all figures (colour and marker), categorical colours from the validated palette
in fixed order (blue, orange, aqua, yellow; checked for colour-vision deficiency), neutral greys for baselines,
Arial 7-8 pt, hairline grids, no top or right spines. Widths follow Frontiers (180 mm full width).
"""
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

DATA, OUT, PLOT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
PLOT.mkdir(parents=True, exist_ok=True)
REV = DATA / "rev1"
DPI = 600
FULL = 7.08  # 180 mm

# palette (dataviz reference instance; slots 1-4 in fixed order) and neutrals
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, DARK = "#8a8984", "#52514e"
CARD, CARD_EDGE = "#f6f5f1", "#c3c2b7"

# one encoding per model, used in every figure
ENC = {
    "gene_name": ("Gene names only", GRAY, "s"),
    "gene_role": ("Gene-annotation flags", GRAY, "X"),
    "kmer3": ("3-mer composition", DARK, "^"),
    "homology": ("MMseqs2 homology", DARK, "s"),
    "domain_retention": ("Domain retention", DARK, "d"),
    "source": ("Data-source features", GRAY, "P"),
    "shortcuts": ("Shortcut features", INK, "v"),
    "esm2": ("ESM-2 650M", BLUE, "o"),
    "esm2_3b": ("ESM-2 3B", BLUE, "8"),
    "prot_t5": ("ProtT5-XL", YELLOW, "h"),
    "fuson": ("FusOn-pLM", AQUA, "D"),
    "fpl": ("FusionPath-like", ORANGE, "p"),
}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "legend.frameon": False, "savefig.dpi": DPI,
    "pdf.fonttype": 42, "lines.solid_capstyle": "round"})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=DPI, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    from PIL import Image  # Frontiers asks for RGB, not RGBA
    Image.open(OUT / f"{name}.png").convert("RGB").save(OUT / f"{name}.png", dpi=(DPI, DPI))


def clean(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def point(ax, x, y, lo, hi, key, horizontal=True, ms=5.6, label=None, lw=1.6):
    _, col, mk = ENC[key]
    err = [[x - lo], [hi - x]] if horizontal else [[y - lo], [hi - y]]
    kw = dict(xerr=err) if horizontal else dict(yerr=err)
    ax.errorbar(x, y, fmt=mk, color=col, ms=ms, mec="white", mew=0.7, elinewidth=lw, capsize=0, zorder=3,
                label=label, **kw)


def handle(key):
    _, col, mk = ENC[key]
    return Line2D([], [], marker=mk, color=col, ls="", ms=6.5, mec="white", mew=0.7)


def panel(ax, letter, x=-0.02):
    ax.text(x, 1.02, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, ha="right", va="bottom")


# ------------------------------------------------------------------ Figure 1: study design
stages = [
    ("1", "Reproduction", "Labels derived from\npartner-gene annotations",
     "Run FusionPath's public\npipeline and characterise\nits labels",
     "9.4% vs 22.2% pathogenic;\n44 of 54 paediatric\ndrivers unlabelled"),
    ("2", "Held-out splits", "Genes shared between\ntraining and test sets",
     "Rows → sequences →\nfusion pairs → whole\ngenes held out",
     "Gene names: 1.000 → 0.500;\npLM embeddings 0.68–0.78\non unseen genes"),
    ("3", "Role-free labels", "Gene-role labels and\ndata source",
     "Cancer evidence vs normal-\ntissue chimeras; both\nclasses from one source",
     "Sequence origin alone: 0.94;\nadding embeddings:\nat most +0.044"),
    ("4", "Paediatric panel", "Coverage of childhood\ncancer drivers",
     "47 driver fusions and\n483 isoforms, scored\nunder both label sets",
     "Gene-role labels: 20–34;\ngene-role-free: 43–45;\ngene names alone: 41"),
]
fig, ax = plt.subplots(figsize=(FULL, 3.15))
ax.set_position([0.005, 0.0, 0.99, 1.0])
ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")
w, gap, x0 = 23.0, 2.33, 0.0
for i, (num, title, shortcut, design, result) in enumerate(stages):
    x = x0 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 5.0), w, 46.5, boxstyle="round,pad=0,rounding_size=1.0", fc=CARD, ec=CARD_EDGE, lw=0.6))
    ax.add_patch(FancyBboxPatch((x, 44.0), w, 7.5, boxstyle="round,pad=0,rounding_size=1.0", fc=BLUE, ec=BLUE, lw=0))
    ax.add_patch(plt.Rectangle((x, 44.0), w, 2.0, fc=BLUE, ec=BLUE, lw=0))
    ax.text(x + 1.3, 47.6, f"{num}  {title}", fontsize=7.2, fontweight="bold", color="white", va="center")
    ax.text(x + 1.3, 41.6, "SHORTCUT ADDRESSED", fontsize=5.0, color=MUTED, va="top", fontweight="bold")
    ax.text(x + 1.3, 38.9, shortcut, fontsize=6.2, color=INK, va="top", linespacing=1.25)
    ax.text(x + 1.3, 30.6, "DESIGN", fontsize=5.0, color=MUTED, va="top", fontweight="bold")
    ax.text(x + 1.3, 27.9, design, fontsize=6.2, color=INK2, va="top", linespacing=1.25)
    ax.plot([x + 1.3, x + w - 1.3], [17.0, 17.0], color=CARD_EDGE, lw=0.5)
    ax.text(x + 1.3, 15.6, "KEY RESULT", fontsize=5.0, color=MUTED, va="top", fontweight="bold")
    ax.text(x + 1.3, 12.9, result, fontsize=6.2, color=BLUE, va="top", linespacing=1.25, fontweight="bold")
    if i < 3:
        ax.add_patch(FancyArrowPatch((x + w + 0.25, 28.0), (x + w + gap - 0.25, 28.0), arrowstyle="-|>",
                                     mutation_scale=6, color=MUTED, lw=0.8))
ax.text(50, 1.8, "At every stage, pLM-based models are compared with baselines that see only the shortcut:\n"
        "gene names, gene roles, protein length, genomic layout and data source", fontsize=6.2, color=INK2,
        ha="center", va="center", linespacing=1.3)
save(fig, "Figure1")
pd.DataFrame([{"stage": f"{s[0]} {s[1]}", "shortcut_addressed": s[2].replace("\n", " "), "design": s[3].replace("\n", " "),
               "result_shown": s[4].replace("\n", "; ")} for s in stages]).to_csv(PLOT / "Figure1_data.csv", index=False)

# ------------------------------------------------------------------ Figure 2: AUROC by split
b = pd.read_csv(REV / "reanalysis/fp.csv")
b = b[(b.labelset == "as_published") & (b.kind == "model")]
splits = [("random_row", "Random rows\n(FusionPath main split)"), ("random_seq", "Random sequences\n(sequence-grouped split)"),
          ("pair_grouped", "Fusion pair\nheld out"), ("genes_held_out", "Genes held out\n(10 assignments)")]
series = [("gene_name", "gene_name"), ("kmer3", "kmer3"), ("esm2_650m/mlp", "esm2"), ("fuson_plm/mlp", "fuson"),
          ("fusionpath_like/mlp", "fpl")]
fig2_rows = []
fig, ax = plt.subplots(figsize=(5.9, 3.3))
for i in range(len(splits)):
    if i % 2 == 1:
        ax.axvspan(i - 0.5, i + 0.5, color="#f3f2ee", zorder=0, lw=0)
for k, (model, key) in enumerate(series):
    for i, (sp, lab) in enumerate(splits):
        r = b[(b.split == sp) & (b.name == model)].iloc[0]
        point(ax, i + (k - 2) * 0.15, r.AUROC, r.ci_lo, r.ci_hi, key, horizontal=False)
        fig2_rows.append({"split": sp, "split_label": lab.replace("\n", " "), "model": ENC[key][0], "result_row_model": model,
                          "AUROC": r.AUROC, "CI_low": r.ci_lo, "CI_high": r.ci_hi, "SD_over_assignments": r.sd_over_assignments,
                          "n_assignments": r.n_assignments, "interval": "gene-resampled bootstrap (1,000)"})
ax.axhline(0.95, color=INK2, lw=0.7, ls=(0, (4, 3)), zorder=1)
ax.text(-0.45, 0.944, "Published FusionPath AUROC (0.95)", ha="left", va="top", fontsize=6.4, color=INK2)
ax.axhline(0.5, color=MUTED, lw=0.6, ls=":", zorder=1)
ax.text(3.47, 0.506, "Chance", ha="right", va="bottom", fontsize=6.4, color=MUTED)
ax.set_xticks(range(len(splits)), [s[1] for s in splits])
ax.set_ylim(0.45, 1.03); ax.set_xlim(-0.5, 3.5)
ax.set_ylabel("AUROC (gene-resampled 95% CI)")
ax.set_xlabel("Overlap between training and test sets removed, from least to most")
clean(ax)
ax.legend([handle(k) for _, k in series], [ENC[k][0] for _, k in series], loc="lower center", bbox_to_anchor=(0.5, 1.0),
          ncol=5, handletextpad=0.1, columnspacing=1.0)
save(fig, "Figure2")
pd.DataFrame(fig2_rows).to_csv(PLOT / "Figure2_data.csv", index=False)

# ------------------------------------------------------------------ Figure 3: held-out genes, every model of Table 3
hb = pd.read_csv(REV / "reanalysis/fp.csv")
hb = hb[(hb.labelset == "as_published") & (hb.kind == "model") & (hb.split == "genes_held_out")].set_index("name")
hr = pd.read_csv(REV / "reanalysis/fpr.csv")
hr = hr[(hr.kind == "model") & (hr.split == "genes_held_out")].set_index("name")
f3 = [("Baselines without a pLM", [("gene_name", hb, "Gene names only", "gene_name"),
                                   ("gene_role", hb, "Gene-annotation flags", "gene_role"),
                                   ("kmer3", hb, "3-mer composition", "kmer3"),
                                   ("homology", hb, "MMseqs2 homology", "homology"),
                                   ("domain_retention", hr, "Domain retention (Pfam)", "domain_retention")]),
      ("MLP on one pLM embedding", [("esm2_650m/mlp", hb, "ESM-2 650M", "esm2"), ("esm2_3b/mlp", hb, "ESM-2 3B", "esm2_3b"),
                                    ("prot_t5/mlp", hb, "ProtT5-XL", "prot_t5"),
                                    ("fuson_plm/mlp", hb, "FusOn-pLM, 1,022-residue windows", "fuson"),
                                    ("fuson_plm_native/mlp", hb, "FusOn-pLM, 2,000-residue windows", "fuson")]),
      ("FusionPath-like model and rebuild", [("fusionpath_like/mlp", hb, "FusionPath-like", "fpl"),
                                             ("fusionpath_rebuild/full", hr, "FusionPath rebuild", "fpl"),
                                             ("fusionpath_rebuild/dom_go", hr, "Rebuild, domains and GO only", "fpl")])]
fig, ax = plt.subplots(figsize=(6.6, 5.0))
plt.rcParams.update({"ytick.labelsize": 8, "xtick.labelsize": 8})
f3_rows, ylabels, y, heads = [], [], 0.0, []
for grp, items in f3:
    heads.append((-y + 0.75, grp))
    for name, src, label, key in items:
        r = src.loc[name]
        point(ax, r.AUROC, -y, r.ci_lo, r.ci_hi, key)
        ax.text(1.065, -y, f"{r.AUROC:.3f} ({r.ci_lo:.3f}–{r.ci_hi:.3f})".replace("-", "−").replace("−0", "−0"),
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7.4, color=INK2)
        ylabels.append((-y, label))
        f3_rows.append({"group": grp, "model": label, "result_row": name, "AUROC": r.AUROC, "CI_low": r.ci_lo,
                        "CI_high": r.ci_hi, "n_assignments": r.n_assignments, "interval": "gene-resampled bootstrap (1,000)"})
        y += 1
    y += 0.9
for yy, grp in heads:
    ax.text(-0.02, yy, grp, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8, fontweight="bold",
            color=INK)
ax.text(1.065, 1.2, "AUROC (95% CI)", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.6,
        fontweight="bold", color=INK)
ax.axvline(0.5, color=MUTED, lw=0.6, ls=":", zorder=1)
ax.text(0.507, 1.2, "Chance", fontsize=6.4, color=MUTED, va="center")
ax.set_yticks([p for p, _ in ylabels], [lab for _, lab in ylabels])
ax.tick_params(axis="y", length=0)
ax.set_xlim(0.40, 1.02); ax.set_ylim(-y + 0.6, 1.6)
ax.set_xlabel("AUROC, gene-disjoint split: both genes unseen (gene-resampled 95% CI)")
clean(ax, "x")
ax.spines["left"].set_visible(False)
save(fig, "Figure3")
pd.DataFrame(f3_rows).to_csv(PLOT / "Figure3_data.csv", index=False)

# ------------------------------------------------------------------ Figure 4: gene-role-free labels
v = pd.read_csv(REV / "reanalysis/v2.csv")
rows = [("v2_seeds50", "all test pairs", "AB_all", "All sources"),
        ("v2_seeds50", "all test pairs", "AB_nonRT", "All sources, read-through\npairs removed"),
        ("v2_seeds50", "both genes unseen", "AB_nonRT", "Same, both genes unseen"),
        ("v2_matched", "all test pairs", "A_fusionpath", "Our translations of\nFusionPath breakpoints"),
        ("v2_matched", "all test pairs", "fusiongdb2", "FusionGDB2"),
        ("v2_matched", "all test pairs", "fusionpath_db", "FusionPath database"),
        ("v2_matched", "all test pairs", "fuson_db", "FusOn-DB")]
models = [("source", "source"), ("shortcuts", "shortcuts"), ("esm2_logreg", "esm2"), ("fuson_logreg", "fuson"),
          ("prot_t5_logreg", "prot_t5")]
gains = [("esm2+shortcuts - shortcuts", "esm2"), ("fuson+shortcuts - shortcuts", "fuson"),
         ("prot_t5+shortcuts - shortcuts", "prot_t5")]


def val(ds, tp, sub, name):
    r = v[(v.dataset == ds) & (v.test_pairs == tp) & (v.subset == sub) & (v.name == name)].iloc[0]
    return r.AUROC, r.ci_lo, r.ci_hi, r.n_assignments, r.median_test_pairs


plt.rcParams.update({"ytick.labelsize": 8, "xtick.labelsize": 8, "axes.labelsize": 8.5, "legend.fontsize": 8})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 5.4), sharey=True, gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.06})
ypos = {r[2] + r[1]: -(i + (0.8 if i >= 3 else 0)) for i, r in enumerate(rows)}
for ax, items, spread in ((a1, models, 0.17), (a2, gains, 0.24)):
    for k, (comp, key) in enumerate(items):
        for ds, tp, sub, _ in rows:
            x, lo, hi, _, _ = val(ds, tp, sub, comp)
            point(ax, x, ypos[sub + tp] + ((len(items) - 1) / 2 - k) * spread, lo, hi, key, ms=5.2, lw=1.5)
for ax in (a1, a2):
    ax.axhspan(-2.6, -10, color="#f3f2ee", zorder=0, lw=0)
    clean(ax, "x")
    ax.tick_params(axis="y", length=0)
a2.axvline(0, color=INK2, lw=0.7, zorder=1)
for xx in (-0.05, 0.05):
    a2.axvline(xx, color=MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1)
a1.set_yticks([ypos[r[2] + r[1]] for r in rows], [r[3] for r in rows])
a1.set_xlim(0.40, 1.0); a1.set_xlabel("AUROC, at least one gene unseen (gene-resampled 95% CI)")
a2.set_xlim(-0.22, 0.36); a2.set_xlabel("Gain from adding the embedding to the\nshortcut features, ΔAUROC (95% CI)")
panel(a1, "A"); panel(a2, "B")
a1.text(0.41, 0.62, "Pooled sources (50 gene assignments)", fontsize=6.4, color=INK2, va="center", fontweight="bold")
a1.text(0.41, -3.05, "Both classes from one source (21 gene assignments)", fontsize=6.4, color=INK2, va="center",
        fontweight="bold")
a2.text(0.058, 0.62, "±0.05 margin", fontsize=6.0, color=MUTED, va="center", ha="left")
a1.set_ylim(min(ypos.values()) - 0.6, 0.95)
a1.legend([handle(k) for _, k in models], [ENC[k][0] for _, k in models], loc="upper center", bbox_to_anchor=(0.5, -0.17),
          ncol=3, handletextpad=0.1, columnspacing=1.0)
a2.legend([handle(k) for _, k in gains], [ENC[k][0] + " + shortcut features" for _, k in gains], loc="upper center",
          bbox_to_anchor=(0.5, -0.17), ncol=1, handletextpad=0.1)
save(fig, "Figure4")
pd.DataFrame([{"panel": pn, "dataset": ds, "test_pairs": tp, "training_set": sub, "training_set_label": lab.replace("\n", " "),
               "quantity": ENC[key][0] + suffix, "value": val(ds, tp, sub, comp)[0], "CI_low": val(ds, tp, sub, comp)[1],
               "CI_high": val(ds, tp, sub, comp)[2], "n_assignments": val(ds, tp, sub, comp)[3],
               "median_test_pairs": val(ds, tp, sub, comp)[4]}
              for pn, items, suffix in (("A: AUROC", models, ""), ("B: gain over shortcut features", gains, " + shortcut features"))
              for comp, key in items for ds, tp, sub, lab in rows]).to_csv(PLOT / "Figure4_data.csv", index=False)

# ------------------------------------------------------------------ Figure 5: paediatric panel
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

fc = pd.read_csv(REV / "fp_rev/panel/fusion_calls.csv")
fp_share = fc.groupby(["fusion", "model"]).called.mean().unstack()
length = fc.groupby("fusion").median_len.first()
ps = pd.read_csv(REV / "v2_strict/panel/panel_scores.csv")
v2_share = (ps.groupby(["model", "repeat", "fusion"]).called.mean() > 0.5).groupby(["fusion", "model"]).mean().unstack()
fpl = pd.read_csv(DATA / "panel_fusionpath_labels.csv").set_index("fusion").fusionpath_label
ov = pd.read_csv(REV / "reanalysis/panel_pretrain_overlap_rev.csv")
seen = ov.assign(t=ov.iso_split == "train").groupby("fusion").t.sum()
dis = pd.read_csv(DATA / "panel_pretrain_overlap_isoforms.csv").groupby("fusion").disease.first()
order = length.sort_values(kind="stable").index
cols = [("FusionPath-style labels", "Gene names", fp_share["gene_name"]),
        ("FusionPath-style labels", "ESM-2 MLP", fp_share["esm2_650m/mlp"]),
        ("FusionPath-style labels", "FusOn-pLM MLP", fp_share["fuson_plm/mlp"]),
        ("FusionPath-style labels", "FusionPath-like", fp_share["fusionpath_like/mlp"]),
        ("Gene-role-free labels", "Gene names", v2_share["gene_name"]),
        ("Gene-role-free labels", "Shortcut features", v2_share["shortcuts"]),
        ("Gene-role-free labels", "ESM-2 MLP", v2_share["esm2_mlp"]),
        ("Gene-role-free labels", "FusOn-pLM MLP", v2_share["fuson_mlp"]),
        ("Gene-role-free labels", "ESM-2 + shortcut\nfeatures", v2_share["esm2+shortcuts"])]
M = np.column_stack([c[2].reindex(order).to_numpy(dtype=float) for c in cols])
cmap = LinearSegmentedColormap.from_list("blue", ["#f3f2ee", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"])
n = len(order)
fig, ax = plt.subplots(figsize=(FULL, 9.6))
gap = 0.45
xs = [0, 1, 2, 3] + [4 + gap + j for j in range(5)]
for j, x in enumerate(xs):
    for i in range(n):
        val_ = M[i, j]
        ax.add_patch(plt.Rectangle((x + 0.03, i + 0.05), 0.94, 0.9, color=cmap(val_), lw=0))
        if val_ < 0.999:
            ax.text(x + 0.5, i + 0.5, f"{val_:.0%}", ha="center", va="center", fontsize=5.2,
                    color="white" if val_ > 0.6 else INK)
ax.axis("off")
for j, (grp, name, _) in enumerate(cols):
    ax.text(xs[j] + 0.5, -0.3, name, ha="center", va="bottom", fontsize=6.4, color=INK2, rotation=90, linespacing=1.1)
for x0_, x1_, title in [(xs[0], xs[3] + 1, "FusionPath-style labels"), (xs[4], xs[8] + 1, "Gene-role-free labels")]:
    ax.plot([x0_ + 0.05, x1_ - 0.05], [-7.6, -7.6], color=INK2, lw=0.7)
    ax.text((x0_ + x1_) / 2, -7.9, title, ha="center", va="bottom", fontsize=7.2, color=INK, fontweight="bold")
xa = xs[-1] + 1.85
side = [(0, "Median length (aa)"), (1.3, "FusionPath-style\nlabel"), (2.45, "In FusOn-pLM\ntraining split")]
for dx, head in side:
    ax.text(xa + dx, -0.3, head, ha="center", va="bottom", fontsize=6.4, color=INK2, rotation=90, linespacing=1.1)
for i, f in enumerate(order):
    ax.text(-0.12, i + 0.5, f, ha="right", va="center", fontsize=6.2, color=INK, fontstyle="italic")
    ax.text(xa, i + 0.5, f"{length[f]:,.0f}", ha="center", va="center", fontsize=5.9, color=INK2)
    lab = fpl.get(f, "")
    ax.text(xa + 1.3, i + 0.5, {"Pathogenic": "P", "Uncertain": "U", "Benign": "B"}.get(lab, ""), ha="center",
            va="center", fontsize=5.9, color=INK if lab == "Pathogenic" else MUTED,
            fontweight="bold" if lab == "Pathogenic" else None)
    ax.text(xa + 2.45, i + 0.5, "●" if seen.get(f, 0) > 0 else "○", ha="center", va="center", fontsize=5.9, color=INK2)
k = int((length.reindex(order) <= 1500).sum())
ax.plot([-3.9, xa + 2.9], [k, k], color=INK2, lw=0.6, ls=(0, (3, 2)))
ax.text(-3.9, k + 0.15, "> 1,500 aa", fontsize=6, color=INK2, va="top")
kx, ky = xs[4], n + 1.0
for s_ in range(6):
    vv = s_ / 5
    ax.add_patch(plt.Rectangle((kx + s_ * 0.66, ky), 0.62, 0.7, color=cmap(vv), lw=0, clip_on=False))
    ax.text(kx + s_ * 0.66 + 0.31, ky + 1.35, f"{vv:.0%}", ha="center", va="center", fontsize=5.6, color=INK2)
ax.text(kx - 0.2, ky + 0.35, "Called in share of\ncalibration repeats", ha="right", va="center", fontsize=6.2, color=INK2)
ax.text(-3.9, ky + 2.8, "Label: P pathogenic, U uncertain (not used for training or testing).   "
        "● at least one isoform in the FusOn-pLM training split; ○ none.", ha="left", va="center", fontsize=5.9, color=INK2)
ax.set_xlim(-4.0, xa + 3.0)
ax.set_ylim(n + 3.6, -10.0)
# full-resolution copy for the Supplement (Supplementary Figure S1), from the same drawing at 1,200 dpi
fig.savefig(OUT / "FigureS1_panel_full_resolution.png", dpi=1200, bbox_inches="tight", pad_inches=0.04, facecolor="white")
save(fig, "Figure5")
f5 = pd.DataFrame(M, columns=[f"{g} | {c.replace(chr(10), ' ')}" for g, c, _ in cols])
f5.insert(0, "fusion", list(order)); f5.insert(1, "disease", [dis.get(f, "") for f in order])
f5.insert(2, "median_length_aa", length.reindex(order).to_numpy())
f5["FusionPath_style_label"] = [fpl.get(f, "") for f in order]
f5["isoforms_in_FusOn_pLM_training"] = [int(seen.get(f, 0)) for f in order]
f5.to_csv(PLOT / "Figure5_data.csv", index=False)  # cells: share of 5 calibration repeats in which the fusion is called
print("figures written to", OUT)
