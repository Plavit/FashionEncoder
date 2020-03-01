#!/usr/bin/env bash

DATASET_ROOT = "/mnt/0/polyvore"
DATASET_FILE = "train_no_dup.json"
TFRECORD_TEMPLATE = "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-{0:03}-{1}.tfrecord"
SHARD_COUNT = 10

python -m "src.data.build_dataset" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-file "${DATASET_FILE}" \
  --tfrecord-template "${TFRECORD_TEMPLATE}" \
  --shard-count 10