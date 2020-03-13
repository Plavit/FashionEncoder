#!/usr/bin/env bash

python -m "src.models.transformer.encoder_main" \
  --dataset-files "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-cleaned-000-10.tfrecord" \
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
  --batch-size 128 \
  --epoch-count 300 \
  --mode "train" \
  --hidden-size 1024 \
  --filter-size 1024 \
  --masking-mode "single-token" \
  --num-heads 32 \
  --valid-batch-size 2 \
  --learning-rate "0.0005" \
  --category-embedding \
  --categories-count 5000


#python -m "src.models.transformer.encoder_main" \
#  --dataset-files "/mnt/0/projects/outfit-generation/data/processed/tfrecords/debug-cleaned-raw-000-1.tfrecord" \
#  --test-files "/mnt/0/projects/outfit-generation/data/processed/tfrecords/valid-cleaned-raw-000-1.tfrecord" \
#  --batch-size 16 \
#  --epoch-count 1 \
#  --mode "train" \
#  --hidden-size 512 \
#  --filter-size 1024 \
#  --masking-mode "single-token" \
#  --num-heads 16 \
#  --valid-batch-size 2 \
#  --with-cnn

#python -m "src.models.transformer.encoder_main" \
#  --dataset-files "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-cleaned-raw-000-10.tfrecord" \
#     "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-cleaned-raw-001-10.tfrecord" \
#     "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-cleaned-raw-002-10.tfrecord" \
#     "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-cleaned-raw-003-10.tfrecord" \
#     "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-cleaned-raw-004-10.tfrecord" \
#     "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-cleaned-raw-005-10.tfrecord" \
#     "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-cleaned-raw-006-10.tfrecord" \
#     "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-cleaned-raw-007-10.tfrecord" \
#     "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-cleaned-raw-008-10.tfrecord" \
#     "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-cleaned-raw-009-10.tfrecord" \
#  --test-files "/mnt/0/projects/outfit-generation/data/processed/tfrecords/valid-cleaned-raw-000-1.tfrecord" \
#  --batch-size 16 \
#  --epoch-count 250 \
#  --mode "train" \
#  --hidden-size 512 \
#  --filter-size 1024 \
#  --masking-mode "single-token" \
#  --num-heads 16 \
#  --valid-batch-size 2 \
#  --with-cnn

