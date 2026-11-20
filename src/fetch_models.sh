#!/bin/bash
# # ProtT5-XL encoder, ProtBERT, ESM-2 3B (Hugging Face) and the MMseqs2 static binary, into ${F2P_MODELS:-${F2P_ROOT}/models}.
set -uo pipefail
D=${F2P_MODELS:-${F2P_ROOT:?set F2P_ROOT}/models}
mkdir -p $D/hf_cache $D/mmseqs
export HF_HOME=$D/hf_cache
PY=${F2P_VENV:-${F2P_ROOT}/.venv}/bin/python
for m in Rostlab/prot_t5_xl_half_uniref50-enc Rostlab/prot_bert facebook/esm2_t36_3B_UR50D; do
  echo "== $m $(date)"
  $PY -c "
from huggingface_hub import snapshot_download
p = snapshot_download('$m', allow_patterns=['*.json','*.txt','*.model','*.safetensors','*.bin','spiece*','vocab*'])
print('ok', p)
" || echo "FAILED $m"
done
echo "== mmseqs $(date)"
cd $D/mmseqs && curl -sSL -o mmseqs-linux-avx2.tar.gz https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz && tar xzf mmseqs-linux-avx2.tar.gz && ./mmseqs/bin/mmseqs version && sha256sum mmseqs-linux-avx2.tar.gz > SHA256SUMS
cat > $D/SOURCE.txt <<TXT
Downloaded $(date -I) for the Fusion2Pheno revision (approved by the user 2026-11-17).
hf_cache/: Hugging Face snapshots of Rostlab/prot_t5_xl_half_uniref50-enc, Rostlab/prot_bert, facebook/esm2_t36_3B_UR50D
mmseqs/: MMseqs2 static Linux AVX2 binary from https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
Remove any of these once the revision results show it is not needed.
TXT
du -sh $D/hf_cache/hub/* $D/mmseqs 2>/dev/null
echo DONE
