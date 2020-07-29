#!/usr/bin/env bash

# Builds Maryland Polyvore with raw images
# Builds the training dataset, validaiton FITB and test FITB

# Build the training dataset
DATASET_ROOT="data/raw/maryland/"
DATASET_FILE="train_no_dup.json"
TFRECORD_TEMPLATE="data/processed/tfrecords/mp-images-train-{0:03}-{1}.tfrecord"

python -m "src.data.build_dataset" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-file "${DATASET_FILE}" \
  --tfrecord-template "${TFRECORD_TEMPLATE}" \
  --shard-count 10


# Build the validaiton FITB
DATASET_FILE="valid_no_dup.json"
OUTPUT_FILE="data/processed/tfrecords/mp-fitb-images-valid.tfrecord"

python -m "src.data.build_fitb" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-file "${DATASET_FILE}" \
  --output-path "${OUTPUT_FILE}" \
  --fitb-file "fill_in_blank_valid.json"


# Build the test FITB
DATASET_FILE="test_no_dup.json"
OUTPUT_FILE="data/processed/tfrecords/mp-fitb-images-test.tfrecord"

python -m "src.data.build_fitb" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-file "${DATASET_FILE}" \
  --output-path "${OUTPUT_FILE}" \
  --fitb-file "fill_in_blank_test.json"
