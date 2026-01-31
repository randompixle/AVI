#!/usr/bin/env bash
set -euo pipefail

# T4-friendly defaults (override via env vars if needed)
export AVI_BLOCK_SIZE=${AVI_BLOCK_SIZE:-256}
export AVI_BATCH_SIZE=${AVI_BATCH_SIZE:-6}
export AVI_N_EMBD=${AVI_N_EMBD:-384}
export AVI_N_HEAD=${AVI_N_HEAD:-8}
export AVI_N_LAYER=${AVI_N_LAYER:-8}

# Train base completion model
PYTHONUNBUFFERED=1 python3 train.py 30000 20000 300000
