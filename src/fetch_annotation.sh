#!/bin/bash
# GENCODE v50 basic GTF + pc_transcripts, HGNC complete set (Singh sequence build).
set -euo pipefail
E=${F2P_ROOT:?set F2P_ROOT}/data/labels_external
D=$(date -u +%Y-%m-%dT%H:%MZ)
B=https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50
get() {  # dir url outfile licence note
  local d=$E/$1; mkdir -p "$d"
  if [ -e "$d/$3" ]; then echo "$d/$3 exists, skipping"; return; fi
  curl -sSL --fail --max-time 1800 -o "$d/$3" "$2"
  (cd "$d"; sha256sum "$3" >> SHA256SUMS)
  printf "url: %s\nretrieved: %s\nlicence: %s\nnote: %s\nbytes: %s\n\n" "$2" "$D" "$4" "$5" "$(stat -c %s "$d/$3")" >> "$d/SOURCE.txt"
  echo "ok $1/$3 $(stat -c %s "$d/$3") bytes"
}
get gencode_v50 $B/gencode.v50.basic.annotation.gtf.gz gencode.v50.basic.annotation.gtf.gz "GENCODE/EMBL-EBI, no restrictions" "GENCODE release 50, GRCh38, basic annotation"
get gencode_v50 $B/gencode.v50.pc_transcripts.fa.gz gencode.v50.pc_transcripts.fa.gz "GENCODE/EMBL-EBI, no restrictions" "GENCODE release 50, protein-coding transcript sequences with UTR/CDS ranges"
curl -s --max-time 120 $B/MD5SUMS | grep -E " gencode\.v50\.(basic\.annotation\.gtf|pc_transcripts\.fa)\.gz$" > $E/gencode_v50/MD5SUMS.expected
(cd $E/gencode_v50; md5sum -c MD5SUMS.expected)
get hgnc https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt hgnc_complete_set.txt "CC0" "HGNC complete set (symbols, previous symbols, aliases, Ensembl IDs)"
