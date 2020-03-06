import tensorflow as tf


def parse_example(raw):
    example = tf.io.parse_single_sequence_example(
        raw, sequence_features={
            "categories": tf.io.FixedLenSequenceFeature([], tf.int64),
            "features": tf.io.FixedLenSequenceFeature(2048, tf.float32)
        })
    return example[1]["features"], example[1]["categories"]


def get_dataset(filenames):
    raw_dataset = tf.data.TFRecordDataset(filenames)
    return raw_dataset.map(parse_example)


def append_targets(features, categories, token_pos):
    return (features, categories, token_pos), features


def add_special_token_positions(features, categories):
    seq_length = tf.shape(categories)[0]
    random_position = tf.random.uniform((1,), minval=0, maxval=seq_length, dtype="int32")
    token_positions = tf.expand_dims(random_position, 0)
    return features, categories, token_positions


def get_training_dataset(filenames, batch_size):
    outfits = get_dataset(filenames)
    outfits = outfits.map(add_special_token_positions)
    outfits = outfits.padded_batch(batch_size, ([None, 2048], [None], [None, 1]))
    return outfits.map(append_targets)