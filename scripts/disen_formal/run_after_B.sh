#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/crossinit-disen-formal
source /root/miniconda3/etc/profile.d/conda.sh
conda activate crossinit
until [ -f runs/disen_formal_v1/matrix_failures.json ]; do sleep 15; done
if python -c 'import json,sys; sys.exit(0 if json.load(open("runs/disen_formal_v1/matrix_failures.json"))==[] else 1)'; then
  python -m scripts.disen_formal.cache text --all-formal > cache_text.log 2>&1
  n=$(ls data/disen_formal/cache/*/text_B320.pt 2>/dev/null | wc -l)
  if [ "$n" -eq 8 ]; then
    python -m scripts.disen_formal.run_matrix \
      --protocol experiments/disen_formal_v1/protocol.json \
      --freeze locks/disen_formal_v1/freeze.json \
      --groups L D A --all-formal --resume > run_LDA.log 2>&1
    touch run_LDA.done
  else
    echo "text cache count=$n" > run_LDA.done
  fi
else
  echo "B matrix had failures" > run_LDA.done
fi
