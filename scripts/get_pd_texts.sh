#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="datasets/text"
mkdir -p "$OUT_DIR"

# Public domain Gutenberg IDs (mix of general knowledge + classics)
IDS=(
  84    # Frankenstein
  98    # A Tale of Two Cities
  1342  # Pride and Prejudice
  1661  # Sherlock Holmes
  174   # Dorian Gray
  2701  # Moby-Dick
  3207  # Leviathan (Hobbes)
  11    # Alice's Adventures in Wonderland
  1080  # A Modest Proposal
  1400  # Great Expectations
  2600  # War and Peace
  4217  # The Federalist Papers
  4300  # Ulysses
  1184  # The Count of Monte Cristo
  32032 # The Origin of Species
)

fetch_one() {
  local id="$1"
  local base="https://www.gutenberg.org/files/${id}"
  local out="${OUT_DIR}/gutenberg_${id}.txt"
  # Try common Gutenberg filename patterns
  if curl -fsSL "${base}/${id}-0.txt" -o "$out"; then
    return 0
  fi
  if curl -fsSL "${base}/${id}.txt" -o "$out"; then
    return 0
  fi
  echo "[warn] failed to download id=${id}" >&2
  return 1
}

for id in "${IDS[@]}"; do
  echo "downloading ${id}..."
  fetch_one "$id" || true
  sleep 0.2
 done

echo "done. cleaning headers..."
python3 scripts/strip_gutenberg.py "$OUT_DIR"
