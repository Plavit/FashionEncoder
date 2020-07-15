#!/usr/bin/env bash

DATASET_ROOT="data/raw/polyvore_outfits/"
DATASET_FILE="data/raw/polyvore_outfits/disjoint/test.json"
OUTPUT_FILE="data/processed/tfrecords/pod-fitb-images-test.tfrecord"

python -m "src.data.build_po_fitb" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-file "${DATASET_FILE}" \
  --output-path "${OUTPUT_FILE}" \
  --fitb-file "data/raw/polyvore_outfits/disjoint/fill_in_blank_test.json"
