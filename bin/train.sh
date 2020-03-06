#!/usr/bin/env bash

python -m "src.models.transformer.encoder_main" \
  --dataset-files "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-000-10.tfrecord" \
   "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-001-10.tfrecord" \
   "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-002-10.tfrecord" \
   "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-003-10.tfrecord" \
   "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-004-10.tfrecord" \
   "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-005-10.tfrecord" \
   "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-006-10.tfrecord" \
   "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-007-10.tfrecord" \
   "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-008-10.tfrecord" \
   "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-009-10.tfrecord" \
  --test-files "/mnt/0/projects/outfit-generation/data/processed/tfrecords/valid-000-1.tfrecord" \
  --batch-size 128 \
  --epoch-count 25 \
  --mode "train" \
  --hidden-size 512 \
  --masking-mode "single-token"