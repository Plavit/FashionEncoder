#!/usr/bin/env bash

#python -m "src.models.encoder.encoder_main" \
#  --mode "train" \
#  --param-set "MP_ADD" \
#  --category-dim 32 \
#  --num-heads 32 \
#  --hidden-size 32 \
#  --filter-size 64 \
#  --num-hidden-layers 2 \
#  --epoch-count 100 \
#  --with-category-grouping False \
#  --batch-size 128
#
#python -m "src.models.encoder.encoder_main" \
#  --mode "train" \
#  --param-set "MP_ADD" \
#  --category-dim 64 \
#  --num-heads 32 \
#  --hidden-size 64 \
#  --filter-size 128 \
#  --num-hidden-layers 2 \
#  --epoch-count 100 \
#  --with-category-grouping False \
#  --batch-size 128
#
#python -m "src.models.encoder.encoder_main" \
#  --mode "train" \
#  --param-set "MP_ADD" \
#  --category-dim 128 \
#  --num-heads 32 \
#  --hidden-size 128 \
#  --filter-size 256 \
#  --num-hidden-layers 2 \
#  --epoch-count 100 \
#  --with-category-grouping False \
#  --batch-size 128

python -m "src.models.encoder.encoder_main" \
  --mode "train" \
  --param-set "PO" \
  --num-heads 32 \
  --category-dim 128 \
  --hidden-size 128 \
  --filter-size 256  \
  --num-hidden-layers 1 \
  --epoch-count 100


#python -m "src.models.encoder.param_tuning"

#python -m "src.models.encoder.encoder_main" \
#  --train-files "data/processed/tfrecords/pod-train-000-1.tfrecord" \
#  --valid-files "data/processed/tfrecords/pod-fitb-features-valid.tfrecord" \
#  --test-file "data/processed/tfrecords/pod-fitb-features-test.tfrecord" \
#  --batch-size 128 \
#  --epoch-count 6 \
#  --mode "train" \
#  --hidden-size 128 \
#  --category-dim 128 \
#  --filter-size 256 \
#  --masking-mode "single-token" \
#  --num-heads 16 \
#  --num-hidden-layers 1 \
#  --learning-rate "0.0005" \
#  --categories-count 5000 \
#  --category-merge "add" \
#  --info "Cross, fixed mask, smoothing" \
#  --category-file "/mnt/0/projects/outfit-generation/data/raw/polyvore_outfits/categories.csv" \
#  --category-embedding \
#  --with-category-grouping \
#  --use-mask-category \
#  --categorywise-train \
#  --early-stop \
#  --early-stop-patience 10 \
#  --early-stop-delta "0.002" \
#  --with-mask-category-embedding \
#  --loss "cross"

#python -m "src.models.encoder.encoder_main" \
#  --train-files "data/processed/tfrecords/pod-images-train-000-10.tfrecord" \
#     "data/processed/tfrecords/pod-images-train-001-10.tfrecord" \
#     "data/processed/tfrecords/pod-images-train-002-10.tfrecord" \
#     "data/processed/tfrecords/pod-images-train-003-10.tfrecord" \
#     "data/processed/tfrecords/pod-images-train-004-10.tfrecord" \
#     "data/processed/tfrecords/pod-images-train-005-10.tfrecord" \
#     "data/processed/tfrecords/pod-images-train-006-10.tfrecord" \
#     "data/processed/tfrecords/pod-images-train-007-10.tfrecord" \
#     "data/processed/tfrecords/pod-images-train-008-10.tfrecord" \
#     "data/processed/tfrecords/pod-images-train-009-10.tfrecord" \
#  --valid-files "data/processed/tfrecords/pod-fitb-images-valid.tfrecord" \
#  --test-files "data/processed/tfrecords/pod-fitb-test-valid.tfrecord" \
#  --batch-size 2 \
#  --epoch-count 50 \
#  --mode "train" \
#  --hidden-size 128 \
#  --category-dim 128 \
#  --filter-size 256 \
#  --masking-mode "single-token" \
#  --num-heads 16 \
#  --valid-batch-size 2 \
#  --num-hidden-layers 1 \
#  --learning-rate "0.00025" \
#  --categories-count 5000 \
#  --category-merge "add" \
#  --target-gradient-from 0 \
#  --info "Low hidden size with images" \
#  --category-file "data/raw/polyvore_outfits/categories.csv" \
#  --category-embedding \
#  --use-mask-category \
#  --with-category-grouping \
#  --with-mask-category-embedding \
#  --categorywise-train \
#  --with-cnn \
#  --with-weights "logs/20200714-084843/tf_ckpts/" \
#  --early-stop \
#  --early-stop-patience 8 \
#  --early-stop-delta "0.002"



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

