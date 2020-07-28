#!/usr/bin/env bash

python -m "src.models.encoder.encoder_main" \
  --mode "train" \
  --param-set "MP_CONCAT" \
  --num-heads 4 \
  --category-dim 64 \
  --hidden-size 128 \
  --filter-size 256  \
  --num-hidden-layers 2 \
  --epoch-count 100
