"""How well does FusOn-DB cover paediatric driver fusions, and how many are >2,000 aa?

FusOn-pLM excluded sequences longer than 2,000 amino acids from training.
This script checks which well-known paediatric driver fusions are in FusOn-DB
and how many of their protein isoforms exceed that limit.

Usage:
    python src/fusondb_paediatric_coverage.py \
        --csv data/raw/fuson_db/FusOn-DB.csv --out results
"""
import argparse
from pathlib import Path

import pandas as pd

LENGTH_LIMIT = 2000  # FusOn-pLM training cut-off

# Hand-curated panel of paediatric driver fusions (head, tail, disease group).
# Draft list for scoping; review before using in a paper.
PANEL = [
    # B-/T-ALL
    ("ETV6", "RUNX1", "B-ALL"), ("TCF3", "PBX1", "B-ALL"), ("TCF3", "HLF", "B-ALL"),
    ("BCR", "ABL1", "B-ALL / CML"), ("EP300", "ZNF384", "B-ALL"), ("TCF3", "ZNF384", "B-ALL"),
    ("PAX5", "JAK2", "B-ALL"), ("EBF1", "PDGFRB", "B-ALL (Ph-like)"), ("MEF2D", "BCL9", "B-ALL"),
    ("PICALM", "MLLT10", "T-ALL / AML"), ("SET", "NUP214", "T-ALL"),
    # KMT2A-rearranged (infant ALL / AML)
    ("KMT2A", "AFF1", "Infant ALL"), ("KMT2A", "MLLT1", "Infant ALL / AML"),
    ("KMT2A", "MLLT3", "AML / ALL"), ("KMT2A", "MLLT10", "AML"), ("KMT2A", "ELL", "AML"),
    ("KMT2A", "AFDN", "AML"), ("KMT2A", "EPS15", "ALL"),
    # Paediatric AML
    ("NUP98", "NSD1", "Paediatric AML"), ("NUP98", "KDM5A", "Paediatric AMKL"),
    ("CBFA2T3", "GLIS2", "Paediatric AMKL"), ("RBM15", "MRTFA", "Infant AMKL"),
    ("RUNX1", "RUNX1T1", "AML"), ("CBFB", "MYH11", "AML"), ("PML", "RARA", "APL"),
    ("DEK", "NUP214", "AML"), ("FUS", "ERG", "AML / Ewing-like"),
    # Lymphoma
    ("NPM1", "ALK", "ALCL"),
    # Sarcomas
    ("EWSR1", "FLI1", "Ewing sarcoma"), ("EWSR1", "ERG", "Ewing sarcoma"),
    ("EWSR1", "WT1", "DSRCT"), ("EWSR1", "ATF1", "Clear-cell sarcoma"),
    ("EWSR1", "NR4A3", "Extraskeletal myxoid chondrosarcoma"), ("FUS", "DDIT3", "Myxoid liposarcoma"),
    ("PAX3", "FOXO1", "Alveolar rhabdomyosarcoma"), ("PAX7", "FOXO1", "Alveolar rhabdomyosarcoma"),
    ("SS18", "SSX1", "Synovial sarcoma"), ("SS18", "SSX2", "Synovial sarcoma"),
    ("CIC", "DUX4", "CIC-rearranged sarcoma"), ("BCOR", "CCNB3", "BCOR sarcoma"),
    ("ASPSCR1", "TFE3", "Alveolar soft part sarcoma"), ("HEY1", "NCOA2", "Mesenchymal chondrosarcoma"),
    ("COL1A1", "PDGFB", "DFSP"), ("ETV6", "NTRK3", "Infantile fibrosarcoma / secretory"),
    ("TPM3", "NTRK1", "NTRK-fused tumours"), ("LMNA", "NTRK1", "NTRK-fused tumours"),
    # CNS
    ("KIAA1549", "BRAF", "Pilocytic astrocytoma"), ("ZFTA", "RELA", "Supratentorial ependymoma"),
    ("YAP1", "MAMLD1", "Supratentorial ependymoma"), ("MYB", "QKI", "Angiocentric glioma"),
    ("FGFR1", "TACC1", "Low-grade glioma"), ("MN1", "BEND2", "Astroblastoma"),
    # Other
    ("DNAJB1", "PRKACA", "Fibrolamellar carcinoma"), ("CRTC1", "MAML2", "Mucoepidermoid carcinoma"),
]

# Old gene symbols that may appear in source databases.
SYNONYMS = {
    "ZFTA": ["C11orf95"], "AFDN": ["MLLT4"], "KMT2A": ["MLL"], "MRTFA": ["MKL1"],
    "KDM5A": ["JARID1A"], "NSD1": [], "AFF1": ["AF4", "MLLT2"],
}


def names(gene):
    return {gene, *SYNONYMS.get(gene, [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv).fillna({"cancers": "", "primary_sources": "", "secondary_sources": ""})
    df["length"] = df["aa_seq"].str.len()
    parts = df["fusiongenes"].str.split("::", n=1, expand=True)
    df["head"], df["tail"] = parts[0].str.strip(), parts[1].str.strip()

    lines = ["# FusOn-DB: paediatric driver-fusion coverage", ""]
    lines += [
        f"- Sequences: {len(df):,}",
        f"- Unique head::tail fusions: {df['fusiongenes'].nunique():,}",
        f"- Sequences > {LENGTH_LIMIT} aa: {(df['length'] > LENGTH_LIMIT).sum():,} "
        f"({(df['length'] > LENGTH_LIMIT).mean():.1%})",
        f"- Length median / 95th pct / max: {df['length'].median():.0f} / "
        f"{df['length'].quantile(0.95):.0f} / {df['length'].max():,}",
        f"- Sequences labelled 'non-cancer' in `cancers`: "
        f"{(df['cancers'].astype(str).str.lower() == 'non-cancer').sum():,}",
        "",
    ]

    rows = []
    for head, tail, disease in PANEL:
        hit = df[df["head"].isin(names(head)) & df["tail"].isin(names(tail))]
        rev = df[df["head"].isin(names(tail)) & df["tail"].isin(names(head))]
        rows.append({
            "fusion": f"{head}::{tail}",
            "disease": disease,
            "n_isoforms": len(hit),
            "n_over_limit": int((hit["length"] > LENGTH_LIMIT).sum()),
            "min_len": hit["length"].min() if len(hit) else None,
            "median_len": hit["length"].median() if len(hit) else None,
            "max_len": hit["length"].max() if len(hit) else None,
            "n_reverse_orientation": len(rev),
            "cancers": "; ".join(sorted(set(hit["cancers"].astype(str))))[:200],
            "sources": "; ".join(sorted(set(hit["primary_sources"].astype(str))))[:120],
        })
    cov = pd.DataFrame(rows)
    cov.to_csv(out / "fusondb_paediatric_coverage.csv", index=False)

    missing = cov[cov["n_isoforms"] == 0]
    all_long = cov[(cov["n_isoforms"] > 0) & (cov["n_over_limit"] == cov["n_isoforms"])]
    some_long = cov[(cov["n_over_limit"] > 0) & (cov["n_over_limit"] < cov["n_isoforms"])]
    lines += [
        f"## Panel of {len(cov)} paediatric driver fusions",
        "",
        f"- Absent from FusOn-DB: **{len(missing)}** "
        f"({', '.join(missing['fusion']) or 'none'})",
        f"- Present but every isoform > {LENGTH_LIMIT} aa (fully excluded from FusOn-pLM training): "
        f"**{len(all_long)}** ({', '.join(all_long['fusion']) or 'none'})",
        f"- Some isoforms > {LENGTH_LIMIT} aa: **{len(some_long)}** "
        f"({', '.join(some_long['fusion']) or 'none'})",
        "",
        cov.drop(columns=["cancers", "sources"]).to_markdown(index=False),
        "",
    ]

    # Partner diversity for promiscuous genes (relevant to held-out-partner splits).
    lines += ["## Partner diversity of promiscuous paediatric genes", ""]
    for gene in ["EWSR1", "FUS", "KMT2A", "NUP98", "ETV6", "NTRK3", "NTRK1", "ALK", "BRAF"]:
        sub = df[(df["head"] == gene) | (df["tail"] == gene)]
        partners = set(sub["tail"][sub["head"] == gene]) | set(sub["head"][sub["tail"] == gene])
        lines.append(
            f"- **{gene}**: {len(sub):,} sequences, {len(partners)} partners, "
            f"{(sub['length'] > LENGTH_LIMIT).sum()} > {LENGTH_LIMIT} aa"
        )

    (out / "fusondb_paediatric_coverage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
