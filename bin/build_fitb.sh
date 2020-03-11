#!/usr/bin/env bash

DATASET_ROOT="/mnt/0/polyvore"
DATASET_FILE="test_no_dup.json"
OUTPUT_FILE="/mnt/0/projects/outfit-generation/data/processed/tfrecords/fitb-features.tfrecord"

python -m "src.data.build_fitb" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-file "${DATASET_FILE}" \
  --output-path "${OUTPUT_FILE}" \
  --fitb-file "fill_in_blank_test.json" \
  --with-features
