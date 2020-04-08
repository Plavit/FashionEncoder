#!/usr/bin/env bash

DATASET_ROOT="/mnt/0/projects/outfit-generation/data/raw/polyvore_outfits/"
DATASET_FILE="/mnt/0/projects/outfit-generation/data/raw/polyvore_outfits/disjoint/valid.json"
OUTPUT_FILE="/mnt/0/projects/outfit-generation/data/processed/tfrecords/pod-fitb-images-valid.tfrecord"

python -m "src.data.build_po_fitb" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-file "${DATASET_FILE}" \
  --output-path "${OUTPUT_FILE}" \
  --fitb-file "/mnt/0/projects/outfit-generation/data/raw/polyvore_outfits/disjoint/fill_in_blank_valid.json"
