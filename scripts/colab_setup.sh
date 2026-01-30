#!/usr/bin/env bash
set -euo pipefail

# Colab setup for T4
python3 -m pip install --upgrade pip
python3 -m pip install sentencepiece datasets transformers

# If using LFS datasets in repo
if command -v git >/dev/null 2>&1; then
  git lfs install || true
  git lfs pull || true
fi
