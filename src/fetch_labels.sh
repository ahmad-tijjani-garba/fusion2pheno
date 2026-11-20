#!/bin/bash
# step-1 downloads for the gene-role-independent label set.
set -euo pipefail
E=${F2P_ROOT:?set F2P_ROOT}/data/labels_external
D=$(date -u +%Y-%m-%dT%H:%MZ)
get() {  # dir url outfile licence note
  local d=$E/$1; mkdir -p "$d"
  if [ -n "$(ls -A "$d")" ]; then echo "$d not empty, skipping"; return; fi
  curl -sSL --fail --max-time 900 -o "$d/$3" "$2"
  (cd "$d"; sha256sum "$3" > SHA256SUMS)
  printf "url: %s\nretrieved: %s\nlicence: %s\nnote: %s\nbytes: %s\n" "$2" "$D" "$4" "$5" "$(stat -c %s "$d/$3")" > "$d/SOURCE.txt"
  echo "ok $1/$3 $(stat -c %s "$d/$3") bytes"
}
get mitelman https://storage.googleapis.com/mitelman-data-files/prod/mitelman_db.zip mitelman_db.zip "CC BY 4.0" "Mitelman Database of Chromosome Aberrations and Gene Fusions in Cancer"
get ctat_fusionlib https://github.com/FusionAnnotator/CTAT_HumanFusionLib/releases/download/v0.3.0/fusion_lib.Mar2021.dat.gz fusion_lib.Mar2021.dat.gz "none stated; internal use only, do not redistribute" "CTAT_HumanFusionLib v0.3.0 (Mar2021)"
get singh2020 https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7038929/supplementaryFiles PMC7038929_supplementaryFiles.zip "CC BY 4.0" "Singh et al. 2020 NAR 48:1764 doi 10.1093/nar/gkz1223; Europe PMC supplementary bundle (PMC direct link returned HTML)"
for f in VariantSummaries MolecularProfileSummaries AcceptedClinicalEvidenceSummaries; do
  get civic_nightly/$f https://civicdb.org/downloads/nightly/nightly-$f.tsv nightly-$f.tsv "CC0 1.0" "CIViC nightly export"
done
