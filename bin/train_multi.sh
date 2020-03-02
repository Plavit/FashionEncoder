#!/usr/bin/env bash

python -m "src.models.transformer.encoder_main" \
  --dataset-files "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-000-10.tfrecord" \
  --batch-size 128 \
  --epoch-count 1000 \
  --mode "train_multi"