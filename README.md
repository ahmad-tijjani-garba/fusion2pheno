# Fusion2Pheno

Code for the study *Can current benchmarks tell whether protein language models learn fusion biology? A leakage-controlled study with a paediatric focus*.

The study asks whether fusion-pathogenicity predictors built on protein language models (pLMs) learn about the fused protein, or whether they recognise gene identity and data provenance. It audits and controls each shortcut in turn and compares pLM-based models with baselines that see only that shortcut:

1. **Reproduction.** Rebuild the FusionPath dataset from its public code (commit `6f0733d`) and characterise its labels.
2. **Leakage-controlled splits.** Random rows, then random sequences, then fusion pairs held out, then whole genes held out.
3. **Gene-role-free labels.** Cancer-evidence fusions vs normal-tissue chimeras, with both classes drawn from one source, plus a provenance- and layout-matched set.
4. **Paediatric known-driver challenge panel.** 54 driver fusions (47 with protein sequences), scored under both label sets.

All confidence intervals resample genes (Equation 2 of the paper); a two-level bootstrap that also resamples gene assignments is included.

## Layout

```
src/       analysis code (Python and shell); every script starts with a docstring describing its inputs and outputs
jobs/      Slurm job scripts used on the clusters (partitions and resources must be adapted)
figures/   make_figures.py: draws Figures 2-5, renders Figure 1 and writes the plotted values
           tikz/Figure1.tex: Figure 1 (study design) in TikZ, with its compiled Figure1.pdf
```

All paths are resolved in `src/f2p_root.py` from environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `F2P_ROOT` | project root holding `data/`, `derived/`, `models/` and `results/` | this repository |
| `F2P_CACHE` | large caches (Pfam files, per-residue embeddings) | `$F2P_ROOT/cache` |
| `MMSEQS` | MMseqs2 binary | `mmseqs` on `PATH` |
| `F2P_VENV` | Python virtual environment (job scripts) | `$F2P_ROOT/.venv` |

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export F2P_ROOT=$PWD
bash src/fetch_annotation.sh   # GENCODE v50, HGNC
bash src/fetch_labels.sh       # label sources for the gene-role-free labels
bash src/fetch_models.sh       # ProtT5-XL, ProtBERT, ESM-2 3B, MMseqs2
```

ESM-2 650M and FusOn-pLM are downloaded from Hugging Face on first use. FusOn-pLM is released under CC BY-NC-ND 4.0, so its embeddings are not redistributed here; regenerate them with the embedding scripts.

## Pipeline

| Stage | Paper section | Scripts |
|---|---|---|
| Rebuild FusionPath's dataset; label the paediatric panel with its rules | 3.2, 4.1 | `setup_rebuild.sh`, `jobs/rebuild_fusionpath.sbatch`, `summarise_rebuild.py`, `label_panel_fusionpath_rules.py` |
| FusionPath-style splits and leakage | 3.3, 4.2 | `make_splits.py`, `train_baselines.py`, `fp_benchmark_rev.py` |
| Gene-role-free labels and translated sequences | 3.4, 3.5 | `build_labels.py`, `build_labels_v2.py`, `translate_breakpoints.py` |
| Embeddings and pooling (mean, partner-aware, junction) | 3.6 | `embed_sequences.py`, `embed_any.py`, `embed_pooled.py`, `junction_coverage_uniprot.py` |
| Sequence models on FusionPath-style labels | 3.7, 4.3 | `train_seq_models.py`, `fp_benchmark_rev.py`, `eval_pooling.py`, `pooling_repeats.py`, `compare_junction_sources.py` |
| FusionPath rebuild and domain-retention baseline | 3.9, 4.3 | `pfam_scan.py`, `export_domain_features.py`, `fpr_train.py`, `reanalysis_gaps.py` |
| Gene-role-free benchmark: pooled, single-source, family split, domains | 3.7, 4.4 | `train_v2.py`, `train_v2_checks.py`, `train_v2_matched.py`, `train_v2_domains.py` |
| Attention-pooling and LoRA probes | 3.8, 4.4 | `probes_v2.py`, `reanalysis_probes.py` |
| Review analyses: stricter labels, missing-partner flags, paralogues, data-source split, reduced and stacked embeddings, probes of what embeddings encode, matched set, AUPRC, two-level bootstrap | 3.7, 3.11, 4.3, 4.4 | `round2_extra.py`, `source_split.py`, `round3_embed.py`, `round3b.py`, `matched_vs_source.py`, `matched_details.py` |
| Paediatric known-driver challenge panel | 3.10, 4.5 | `make_rev_inputs.py`, `score_panel.py`, `score_panel_v2.py`, `fp_panel_rev.py`, `panel_pretrain_overlap.py`, `panel_partners.py`, `fusondb_paediatric_coverage.py` |
| Gene-resampled intervals, length bins, provenance of reported numbers | 3.11 | `reanalysis.py`, `bootstrap_cis.py`, `reanalysis_length.py`, `provenance.py` |
| Figures | — | `figures/make_figures.py`, `figures/tikz/Figure1.tex` |

Most scripts that train models were run as Slurm jobs; `jobs/rev_cpu.sbatch` and `jobs/rev_job.sbatch` run any command line, for example:

```bash
sbatch -J v2 jobs/rev_cpu.sbatch "python -u train_v2.py --seeds $(seq -s ' ' 1 50) --subsets AB_all AB_nonRT"
```

To rebuild Figure 1 after editing it, compile `figures/tikz/Figure1.tex` with any LaTeX engine (for example `tectonic Figure1.tex` or `pdflatex Figure1.tex`) and rerun `figures/make_figures.py`.

## Data

The input data come from their original providers and are subject to their terms: FusionPath code and inputs, FusionGDB2, ChimerDB 4.0, CIViC, the Network of Cancer Genes, ClinGen, the Mitelman database, the CTAT fusion library, Singh et al. (2020), FusOn-DB, GENCODE v50, MANE, UniProtKB and HGNC. Files that may not be redistributed, such as the CTAT fusion library, are regenerated by the scripts from the original downloads. Derived labels, split assignments, predictions and result tables will be archived on Zenodo; the DOI will be added here.

## Software

Python 3.12, PyTorch 2.15, Transformers 5.23, scikit-learn 1.9.1, pandas 3.1.1, NumPy 2.5, SciPy 1.18, PyHMMER 0.12 and MMseqs2. Exact versions are in `requirements.txt`.

## Citation

Citation details will be added on publication.
