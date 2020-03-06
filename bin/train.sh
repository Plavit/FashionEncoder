#!/usr/bin/env bash

python -m "src.models.transformer.encoder_main" \
  --dataset-files "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-000-10.tfrecord" \
   "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-cleaned-001-10.tfrecord" \
   "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-cleaned-002-10.tfrecord" \
   "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-cleaned-003-10.tfrecord" \
   "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-cleaned-004-10.tfrecord" \
   "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-cleaned-005-10.tfrecord" \
   "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-cleaned-006-10.tfrecord" \
   "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-cleaned-007-10.tfrecord" \
   "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-cleaned-008-10.tfrecord" \
   "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-cleaned-009-10.tfrecord" \
  --test-files "/mnt/0/projects/outfit-generation/data/processed/tfrecords/valid-cleaned-000-1.tfrecord" \
  --batch-size 8 \
  --epoch-count 50 \
  --mode "train" \
  --hidden-size 512 \
  --masking-mode "single-token" \
  --num-heads 8