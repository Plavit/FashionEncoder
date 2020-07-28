#!/usr/bin/env bash

# Builds Polyvore Outfits Non-Disjoint with raw images
# Builds the training dataset, validaiton FITB and test FITB

# Build the training dataset
DATASET_ROOT="data/raw/polyvore_outfits/"
DATASET_FILE="data/raw/polyvore_outfits/nondisjoint/train.json"
TFRECORD_TEMPLATE="data/processed/tfrecords/po-images-train-{0:03}-{1}.tfrecord"

python -m "src.data.build_po_dataset" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-filepath "${DATASET_FILE}" \
  --tfrecord-template "${TFRECORD_TEMPLATE}" \
  --shard-count 10


# Build the validation FITB
DATASET_FILE="data/raw/polyvore_outfits/nondisjoint/valid.json"
OUTPUT_FILE="data/processed/tfrecords/po-fitb-images-valid.tfrecord"

python -m "src.data.build_po_fitb" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-file "${DATASET_FILE}" \
  --output-path "${OUTPUT_FILE}" \
  --fitb-file "data/raw/polyvore_outfits/nondisjoint/fill_in_blank_valid.json"


# Build the test FITB
DATASET_FILE="data/raw/polyvore_outfits/nondisjoint/test.json"
OUTPUT_FILE="data/processed/tfrecords/po-fitb-images-test.tfrecord"

python -m "src.data.build_po_fitb" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-file "${DATASET_FILE}" \
  --output-path "${OUTPUT_FILE}" \
  --fitb-file "data/raw/polyvore_outfits/nondisjoint/fill_in_blank_test.json"