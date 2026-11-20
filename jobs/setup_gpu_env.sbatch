#!/bin/bash
#SBATCH --job-name=f2p_gpu_setup
#SBATCH --partition=gpu3060,gpua4000,gpu4070ti,gpul40,gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# Partitions, GPUs and resources are those of the cluster used for the paper; adapt them to yours.
# install torch + transformers; download ESM-2 650M and FusOn-pLM weights.
set -euo pipefail
P=${F2P_ROOT:?set F2P_ROOT}
export HF_HOME=$P/models/hf_cache
mkdir -p "$HF_HOME"
source ${F2P_VENV:-${F2P_ROOT}/.venv}/bin/activate
cd "$P"

echo "== node: $(hostname)"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
pip install -q torch transformers
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('transformers', transformers.__version__)"

python - <<'EOF'
from huggingface_hub import snapshot_download
for repo in ["facebook/esm2_t33_650M_UR50D", "ChatterjeeLab/FusOn-pLM"]:
    path = snapshot_download(repo)
    print("downloaded", repo, "->", path)
EOF
du -sh "$HF_HOME"/hub/* 2>/dev/null

echo "== smoke test (inference on 2 sequences per model)"
python - <<'EOF'
import torch
from transformers import AutoModel, AutoTokenizer
for repo in ["facebook/esm2_t33_650M_UR50D", "ChatterjeeLab/FusOn-pLM"]:
    tok = AutoTokenizer.from_pretrained(repo)
    m = AutoModel.from_pretrained(repo, torch_dtype=torch.float16).cuda().eval()
    with torch.no_grad():
        enc = tok(["MKTAYIAKQRQISFVKSHFSRQ", "MSESRGLRLPQLEPDPGHSVLGG"], return_tensors="pt", padding=True).to("cuda")
        h = m(**enc).last_hidden_state
    print(repo, "hidden", tuple(h.shape), "finite", bool(torch.isfinite(h).all()))
EOF
pip freeze | grep -i -E "^(torch|transformers|tokenizers|huggingface)"
