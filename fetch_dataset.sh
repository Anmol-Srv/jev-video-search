#!/usr/bin/env bash
# Fetch MSR-VTT. Only needed for the dashboard (which plays the clips) or to rebuild
# the index from scratch; eval.py and compare.py run off the committed index/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
BASE=https://huggingface.co/datasets/friedrichor/MSR-VTT/resolve/main

[ -f data/MSRVTT_data.json ] || {
  echo "annotations (22 MB)…"; curl -fL# -o data/MSRVTT_data.json "$BASE/raw_data/MSRVTT_data.json"; }

[ -d data/clips_raw/video ] || {
  echo "clips (2.1 GB)…"
  curl -fL# -o data/MSRVTT_Videos.zip "$BASE/MSRVTT_Videos.zip"
  unzip -q -o data/MSRVTT_Videos.zip -d data/clips_raw
  rm -f data/MSRVTT_Videos.zip
}
echo "clips on disk: $(find data/clips_raw -name '*.mp4' | wc -l | tr -d ' ')"
echo "index already committed; rebuild only if you want to: python ingest.py --from-captions data/MSRVTT_data.json --holdout data/msrvtt_test_1k.json"
