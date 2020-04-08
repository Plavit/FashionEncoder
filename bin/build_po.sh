#!/usr/bin/env bash

DATASET_ROOT="/mnt/0/projects/outfit-generation/data/raw/polyvore_outfits/"
DATASET_FILE="/mnt/0/projects/outfit-generation/data/raw/polyvore_outfits/disjoint/train.json"
TFRECORD_TEMPLATE="/mnt/0/projects/outfit-generation/data/processed/tfrecords/pod-images-train-{0:03}-{1}.tfrecord"

python -m "src.data.build_po_dataset" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-filepath "${DATASET_FILE}" \
  --tfrecord-template "${TFRECORD_TEMPLATE}" \
  --shard-count 10