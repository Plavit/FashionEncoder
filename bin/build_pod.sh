#!/usr/bin/env bash

# Builds Polyvore Outfits Disjoint with features extracted via CNN
# Builds the training dataset, validaiton FITB and test FITB

# Build the training dataset
DATASET_ROOT="data/raw/polyvore_outfits/"
DATASET_FILE="data/raw/polyvore_outfits/disjoint/train.json"
TFRECORD_TEMPLATE="data/processed/tfrecords/pod-features-train-{0:03}-{1}.tfrecord"

python -m "src.data.build_po_dataset" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-filepath "${DATASET_FILE}" \
  --tfrecord-template "${TFRECORD_TEMPLATE}" \
  --shard-count 1 \
  --with-features


# Build the validation FITB
DATASET_FILE="data/raw/polyvore_outfits/disjoint/valid.json"
OUTPUT_FILE="data/processed/tfrecords/pod-fitb-features-valid.tfrecord"

python -m "src.data.build_po_fitb" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-file "${DATASET_FILE}" \
  --output-path "${OUTPUT_FILE}" \
  --fitb-file "data/raw/polyvore_outfits/disjoint/fill_in_blank_valid.json" \
  --with-features


# Build the test FITB
DATASET_FILE="data/raw/polyvore_outfits/disjoint/test.json"
OUTPUT_FILE="data/processed/tfrecords/pod-fitb-features-test.tfrecord"

python -m "src.data.build_po_fitb" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-file "${DATASET_FILE}" \
  --output-path "${OUTPUT_FILE}" \
  --fitb-file "data/raw/polyvore_outfits/disjoint/fill_in_blank_test.json" \
  --with-features