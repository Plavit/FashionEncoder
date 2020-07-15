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
    "dense_regularization": 0.001,
    "enc_regularization": 0,

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
        "valid_files": ["data/processed/tfrecords/valid-000-1.tfrecord"],
        "valid_mode": "masking",
        "batch_size": 128,
        "learning_rate": 0.0005,
        "layer_postprocess_dropout": 0.2,
        "dense_regularization": 0.0005,
        "attention_dropout": 0.05,
        "relu_dropout": 0.05,
        "emb_dropout": 0.1
    }
)

MP_CATEGORY = MP.copy()
MP_CATEGORY.update(
    {
        "layer_postprocess_dropout": 0.2,
        "dense_regularization": 0.0005,
        "attention_dropout": 0.05,
        "relu_dropout": 0.05,
        "emb_dropout": 0.1
    }
)

MP_ADD = MP.copy()
MP_ADD.update(
    {
        "category_embedding": True,
        "category_merge": "add",
        "categories_count": 5000,
     }
)

MP_MUL = MP.copy()
MP_MUL.update(
    {
        "category_embedding": True,
        "category_merge": "multiply",
        "categories_count": 5000,
     }
)

MP_CONCAT = MP.copy()
MP_CONCAT.update(
    {
        "category_embedding": True,
        "category_merge": "concat",
        "categories_count": 5000,
     }
)

DISTANCE_BASE = BASE.copy()
DISTANCE_BASE.update(
    loss="distance",
    margin="0.5"
)