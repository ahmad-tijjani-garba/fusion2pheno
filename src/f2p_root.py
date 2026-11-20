"""Locations used by every script. Set them with environment variables:

F2P_ROOT   project root holding data/, results/, derived/ and models/ (default: this repository)
F2P_CACHE  large caches: Pfam files and per-residue embeddings (default: $F2P_ROOT/cache)
MMSEQS     path to the MMseqs2 binary (default: "mmseqs" on PATH)
"""
import os
from pathlib import Path

F2P_ROOT = Path(os.environ.get("F2P_ROOT", Path(__file__).resolve().parents[1]))
F2P_CACHE = Path(os.environ.get("F2P_CACHE", F2P_ROOT / "cache"))
MMSEQS = Path(os.environ.get("MMSEQS", "mmseqs"))
