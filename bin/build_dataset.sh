#!/usr/bin/env bash

DATASET_ROOT="/mnt/0/polyvore"
DATASET_FILE="valid_no_dup.json"
TFRECORD_TEMPLATE="/mnt/0/projects/outfit-generation/data/processed/tfrecords/valid-cleaned-raw-{0:03}-{1}.tfrecord"

python -m "src.data.build_dataset" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-file "${DATASET_FILE}" \
  --tfrecord-template "${TFRECORD_TEMPLATE}" \
  --shard-count 1