#!/usr/bin/env bash

python -m "src.models.transformer.encoder_main" \
  --dataset-files "/mnt/0/projects/outfit-generation/data/processed/tfrecords/pod-train-000-1.tfrecord" \
  --test-files "/mnt/0/projects/outfit-generation/data/processed/tfrecords/valid-000-1.tfrecord" \
  --fitb-file "/mnt/0/projects/outfit-generation/data/processed/tfrecords/pod-fitb-features-valid.tfrecord" \
  --batch-size 128 \
  --epoch-count 200 \
  --mode "train" \
  --hidden-size 512 \
  --filter-size 1024 \
  --masking-mode "single-token" \
  --num-heads 16 \
  --valid-batch-size 2 \
  --num-hidden-layers 1 \
  --learning-rate "0.0005" \
  --categories-count 100 \
  --category-merge "add" \
  --target-gradient-from 40 \
  --info "PreprocessorV2" \
  --category-file "/mnt/0/projects/outfit-generation/data/raw/polyvore_outfits/categories.csv" \
  --use-mask-category \
  --with-category-grouping \
  --with-mask-category-embedding \
  --categorywise-train

#python -m "src.models.transformer.encoder_main" \
#  --dataset-files "/mnt/0/projects/outfit-generation/data/processed/tfrecords/pod-images-train-000-10.tfrecord" \
#     "/mnt/0/projects/outfit-generation/data/processed/tfrecords/pod-images-train-001-10.tfrecord" \
#     "/mnt/0/projects/outfit-generation/data/processed/tfrecords/pod-images-train-002-10.tfrecord" \
#     "/mnt/0/projects/outfit-generation/data/processed/tfrecords/pod-images-train-003-10.tfrecord" \
#     "/mnt/0/projects/outfit-generation/data/processed/tfrecords/pod-images-train-004-10.tfrecord" \
#     "/mnt/0/projects/outfit-generation/data/processed/tfrecords/pod-images-train-005-10.tfrecord" \
#     "/mnt/0/projects/outfit-generation/data/processed/tfrecords/pod-images-train-006-10.tfrecord" \
#     "/mnt/0/projects/outfit-generation/data/processed/tfrecords/pod-images-train-007-10.tfrecord" \
#     "/mnt/0/projects/outfit-generation/data/processed/tfrecords/pod-images-train-008-10.tfrecord" \
#     "/mnt/0/projects/outfit-generation/data/processed/tfrecords/pod-images-train-009-10.tfrecord" \
#  --test-files "/mnt/0/projects/outfit-generation/data/processed/tfrecords/valid-000-1.tfrecord" \
#  --fitb-file "/mnt/0/projects/outfit-generation/data/processed/tfrecords/pod-fitb-images-valid.tfrecord" \
#  --batch-size 10 \
#  --epoch-count 50 \
#  --mode "train" \
#  --hidden-size 512 \
#  --filter-size 1024 \
#  --masking-mode "single-token" \
#  --num-heads 16 \
#  --valid-batch-size 2 \
#  --num-hidden-layers 1 \
#  --learning-rate "0.0005" \
#  --categories-count 100 \
#  --category-merge "add" \
#  --target-gradient-from 40 \
#  --info "PreprocessorV2" \
#  --category-file "/mnt/0/projects/outfit-generation/data/raw/polyvore_outfits/categories.csv" \
#  --category-embedding \
#  --use-mask-category \
#  --with-category-grouping \
#  --with-mask-category-embedding \
#  --categorywise-train \
#  --with-cnn \
#  --checkpoint-dir "/mnt/0/projects/outfit-generation/logs/20200408-cnn-tuning/tf_ckpts"


#python -m "src.models.transformer.encoder_main" \
#  --dataset-files "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-000-10.tfrecord" \
#     "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-001-10.tfrecord" \
#     "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-002-10.tfrecord" \
#     "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-003-10.tfrecord" \
#     "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-004-10.tfrecord" \
#     "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-005-10.tfrecord" \
#     "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-006-10.tfrecord" \
#     "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-007-10.tfrecord" \
#     "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-008-10.tfrecord" \
#     "/mnt/0/projects/outfit-generation/data/processed/tfrecords/train-009-10.tfrecord" \
#  --test-files "/mnt/0/projects/outfit-generation/data/processed/tfrecords/valid-000-1.tfrecord" \
#  --fitb-file "/mnt/0/projects/outfit-generation/data/processed/tfrecords/fitb-valid.tfrecord" \
#  --batch-size 128 \
#  --epoch-count 200 \
#  --mode "train" \
#  --hidden-size 512 \
#  --filter-size 512 \
#  --masking-mode "single-token" \
#  --num-heads 16 \
#  --valid-batch-size 2 \
#  --num-hidden-layers 1 \
#  --learning-rate "0.0005" \
#  --categories-count 10 \
#  --with-category-grouping \
#  --category-embedding \
#  --category-merge "add" \
#  --target-gradient-from 60 \
#  --info "Only one dense layer"

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
#  --fitb-file "/mnt/0/projects/outfit-generation/data/processed/tfrecords/fitb-images.tfrecord" \
#  --batch-size 16 \
#  --epoch-count 250 \
#  --mode "train" \
#  --hidden-size 512 \
#  --filter-size 1024 \
#  --masking-mode "single-token" \
#  --num-heads 16 \
#  --valid-batch-size 2 \
#  --learning-rate "0.0005" \
#  --with-cnn \
#  --target-gradient-from 20

