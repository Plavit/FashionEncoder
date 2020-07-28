#!/usr/bin/env bash

# Builds Maryland Polyvore with features extracted via CNN
# Builds the training dataset, validaiton FITB and test FITB

# Build the training dataset
DATASET_ROOT="data/raw/maryland/"
DATASET_FILE="train_no_dup.json"
TFRECORD_TEMPLATE="data/processed/tfrecords/mp-features-train-{0:03}-{1}.tfrecord"

python -m "src.data.build_dataset" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-file "${DATASET_FILE}" \
  --tfrecord-template "${TFRECORD_TEMPLATE}" \
  --shard-count 1 \
  --with-features


# Build the validaiton FITB
DATASET_FILE="valid_no_dup.json"
OUTPUT_FILE="data/processed/tfrecords/mp-fitb-features-valid.tfrecord"

python -m "src.data.build_fitb" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-file "${DATASET_FILE}" \
  --output-path "${OUTPUT_FILE}" \
  --fitb-file "fill_in_blank_valid.json" \
  --with-features


# Build the test FITB
DATASET_FILE="test_no_dup.json"
OUTPUT_FILE="data/processed/tfrecords/mp-fitb-features-test.tfrecord"

python -m "src.data.build_fitb" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-file "${DATASET_FILE}" \
  --output-path "${OUTPUT_FILE}" \
  --fitb-file "fill_in_blank_test.json" \
  --with-features
