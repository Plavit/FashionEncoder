import tensorflow as tf


def parse_example(raw):
    example = tf.io.parse_single_sequence_example(
          raw, sequence_features={
              "categories": tf.io.FixedLenSequenceFeature([], tf.int64),
              "features": tf.io.FixedLenSequenceFeature(2048, tf.float32)
          })
    return example[1]["features"]


def get_dataset(filenames):
    raw_dataset = tf.data.TFRecordDataset(filenames)
    return raw_dataset.map(parse_example)


def duplicate(example):
    return example, example


def get_training_dataset(filenames, batch_size):
    outfits = get_dataset(filenames)
    outfits = outfits.padded_batch(batch_size, (None, 2048))
    return outfits.map(duplicate)
