#!/usr/bin/env bash

DATASET_ROOT="data/raw/polyvore_outfits/"
DATASET_FILE="data/raw/polyvore_outfits/nondisjoint/train.json"
TFRECORD_TEMPLATE="data/processed/tfrecords/po-features-train-{0:03}-{1}.tfrecord"

python -m "src.data.build_po_dataset" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-filepath "${DATASET_FILE}" \
  --tfrecord-template "${TFRECORD_TEMPLATE}" \
  --shard-count 1