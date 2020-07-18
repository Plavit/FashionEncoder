BASE = {
    "feature_dim": 2048,
    "dtype": "float32",
    "hidden_size": 128,
    "num_hidden_layers": 1,
    "num_heads": 16,
    "filter_size": 256,
    "layer_postprocess_dropout": 0.1,
    "attention_dropout": 0.1,
    "relu_dropout": 0.1,
    "learning_rate": 0.0005,
    "category_dim": 128,
    "batch_size": 128,
    "epoch_count": 200,
    "masking_mode": "single-token",
    "categories_count": 50,
    "category_merge": "add",
    "category_embedding": True,
    "use_mask_category": True,
    "with_category_grouping": True,
    "with_mask_category_embedding": True,
    "categorywise_train": True,
    "early_stop_patience": 10,
    "early_stop_delta": 0.002,
    "early_stop_warmup": 25,
    "early_stop": True,
    "with_cnn": False,
    "target_gradient_from": 0,
    "loss": "cross",
    "valid_mode": "fitb",
    "dense_regularization": 0,
    "enc_regularization": 0,
    "emb_dropout": 0,
    "i_dense_dropout": 0.1
}


MP = BASE.copy()
MP.update(
    {
        "category_embedding": False,
        "with_mask_category_embedding": False,
        "categorywise_train": False,
        "train_files": [
         "data/processed/tfrecords/train-000-10.tfrecord",
         "data/processed/tfrecords/train-001-10.tfrecord",
         "data/processed/tfrecords/train-002-10.tfrecord",
         "data/processed/tfrecords/train-003-10.tfrecord",
         "data/processed/tfrecords/train-004-10.tfrecord",
         "data/processed/tfrecords/train-005-10.tfrecord",
         "data/processed/tfrecords/train-006-10.tfrecord",
         "data/processed/tfrecords/train-007-10.tfrecord",
         "data/processed/tfrecords/train-008-10.tfrecord",
         "data/processed/tfrecords/train-009-10.tfrecord"
        ],
        "test_files": ["data/processed/tfrecords/fitb-features.tfrecord"],
        "valid_files": ["data/processed/tfrecords/fitb-features-valid.tfrecord"],
        "valid_mode": "fitb",
        "batch_size": 96,
        "learning_rate": 0.002,
        "layer_postprocess_dropout": 0.1,
        "i_dense_dropout": 0.1,
        "dense_regularization": 0,
        "attention_dropout": 0.1,
        "relu_dropout": 0.1,
        "emb_dropout": 0,
    }
)

MP_CATEGORY = MP.copy()
MP_CATEGORY.update(
    {
        "category_embedding": True,
        "with_category_grouping": False,
        "categories_count": 5000
    }
)

MP_ADD = MP_CATEGORY.copy()
MP_ADD.update(
    {
        "category_merge": "add",
    }
)

MP_MUL = MP_CATEGORY.copy()
MP_MUL.update(
    {
        "category_merge": "multiply",
    }
)

MP_CONCAT = MP_CATEGORY.copy()
MP_CONCAT.update(
    {
        "category_merge": "concat",
    }
)

MP_NEG = MP_MUL.copy()
MP_NEG.update(
    {
        "test_files": ["data/processed/tfrecords/fitb-features-test-neg.tfrecord"],
        "valid_files": ["data/processed/tfrecords/fitb-features-valid-neg.tfrecord"],
        "valid_mode": "fitb",
    }
)


PO = BASE.copy()
PO.update({
    "train_files": [
        "data/processed/tfrecords/pod-train-000-1.tfrecord"
    ],
    "test_files": ["data/processed/tfrecords/pod-fitb-features-test.tfrecord"],
    "valid_files": ["data/processed/tfrecords/pod-fitb-features-valid.tfrecord"],
    "category_file": "data/raw/polyvore_outfits/categories.csv",
    "categorywise_train": True,
    "category_embedding": False,
    "valid_mode": "fitb",
    "batch_size": 96,
    "learning_rate": 0.002,
    "layer_postprocess_dropout": 0.1,
    "i_dense_dropout": 0.1,
    "dense_regularization": 0,
    "attention_dropout": 0.1,
    "relu_dropout": 0.1,
    "emb_dropout": 0,
})

PO_CATEGORY = PO.copy()
PO_CATEGORY.update({
    "category_embedding": True,
    "with_mask_category_embedding": True,
})

PO_ADD = PO_CATEGORY.copy()
PO_ADD.update({
    "category_merge": "add"
})

PO_MUL = PO_CATEGORY.copy()
PO_MUL.update({
    "category_merge": "mul"
})

PO_CONCAT = PO_CATEGORY.copy()
PO_CONCAT.update({
    "category_merge": "concat"
})

DISTANCE_BASE = BASE.copy()
DISTANCE_BASE.update(
    loss="distance",
    margin="5"
)