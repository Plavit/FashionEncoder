#!/usr/bin/env bash

#python -m "src.models.encoder.param_tuning"

python -m "src.models.encoder.encoder_main" \
  --dataset-files "/mnt/0/projects/outfit-generation/data/processed/tfrecords/pod-train-000-1.tfrecord" \
  --test-files "/mnt/0/projects/outfit-generation/data/processed/tfrecords/valid-000-1.tfrecord" \
  --fitb-file "/mnt/0/projects/outfit-generation/data/processed/tfrecords/pod-fitb-features-valid.tfrecord" \
  --batch-size 128 \
  --epoch-count 200 \
  --mode "train" \
  --hidden-size 64 \
  --category-dim 64 \
  --filter-size 64 \
  --masking-mode "category-masking" \
  --num-heads 8 \
  --num-hidden-layers 3 \
  --learning-rate "0.0005" \
  --categories-count 5000 \
  --category-merge "add" \
  --info "Cross, fixed mask" \
  --category-file "/mnt/0/projects/outfit-generation/data/raw/polyvore_outfits/categories.csv" \
  --category-embedding \
  --with-category-grouping \
  --use-mask-category \
  --categorywise-train \
  --early-stop \
  --early-stop-patience 10 \
  --early-stop-delta "0.002" \
  --loss "cross"

#python -m "src.models.encoder.encoder_main" \
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
#  --hidden-size 128 \
#  --category-dim 128 \
#  --filter-size 128 \
#  --masking-mode "single-token" \
#  --num-heads 16 \
#  --valid-batch-size 2 \
#  --num-hidden-layers 1 \
#  --learning-rate "0.0005" \
#  --categories-count 50 \
#  --category-merge "add" \
#  --target-gradient-from 0 \
#  --info "Low hidden size with images" \
#  --category-file "/mnt/0/projects/outfit-generation/data/raw/polyvore_outfits/categories.csv" \
#  --category-embedding \
#  --use-mask-category \
#  --with-category-grouping \
#  --with-mask-category-embedding \
#  --categorywise-train \
#  --with-cnn


#python -m "src.models.encoder.encoder_main" \
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
#  --fitb-file "/mnt/0/projects/outfit-generation/data/processed/tfrecords/fitb-features.tfrecord" \
#  --batch-size 128 \
#  --epoch-count 200 \
#  --mode "train" \
#  --hidden-size 128 \
#  --category-dim 128 \
#  --filter-size 128 \
#  --masking-mode "single-token" \
#  --num-heads 16 \
#  --valid-batch-size 2 \
#  --num-hidden-layers 1 \
#  --learning-rate "0.0005" \
#  --categories-count 5000 \
#  --target-gradient-from 30 \
#  --category-embedding \
#  --with-category-grouping \
#  --category-merge "add" \
#  --target-gradient-from 0 \
#  --info "Polyvore Maryland - with model" \
#  --early-stop \
#  --early-stop-patience 6 \
#  --early-stop-delta "0.002"

#python -m "src.models.encoder.encoder_main" \
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

#python -m "src.models.encoder.encoder_main" \
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

