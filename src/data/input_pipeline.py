import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import array_to_img, img_to_array
from PIL import Image
import numpy as np


def parse_example_with_features(raw):
    example = tf.io.parse_single_sequence_example(
        raw, sequence_features={
            "categories": tf.io.FixedLenSequenceFeature([], tf.int64),
            "features": tf.io.FixedLenSequenceFeature(2048, tf.float32)
        })
    return example[1]["features"], example[1]["categories"]


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    return tf.keras.applications.inception_v3.preprocess_input(tf.image.resize(img, [299, 299]))


def parse_example_with_images(raw):
    example = tf.io.parse_single_sequence_example(
        raw, sequence_features={
            "categories": tf.io.FixedLenSequenceFeature([], tf.int64),
            "images": tf.io.FixedLenSequenceFeature([], tf.string)
        })

    raw_imgs = example[1]["images"]
    images = tf.map_fn(decode_img, raw_imgs, dtype=tf.float32)
    return images, example[1]["categories"]


def get_dataset(filenames, with_features):
    raw_dataset = tf.data.TFRecordDataset(filenames)
    if with_features:
        return raw_dataset.map(parse_example_with_features)
    else:
        return raw_dataset.map(parse_example_with_images)


def append_targets(features, categories, token_pos):
    return (features, categories, token_pos), features


def add_random_mask_positions(features, categories):
    seq_length = tf.shape(categories)[0]
    random_position = tf.random.uniform((1,), minval=0, maxval=seq_length, dtype="int32")
    token_positions = tf.expand_dims(random_position, 0)
    return features, categories, token_positions


def get_training_dataset(filenames, batch_size, with_features):
    outfits = get_dataset(filenames, with_features).cache()
    outfits = outfits.map(add_random_mask_positions, tf.data.experimental.AUTOTUNE)
    outfits = outfits.shuffle(3000, 123)

    if with_features:
        outfits = outfits.padded_batch(batch_size, ([None, 2048], [None], [None, 1]), drop_remainder=True)
    else:
        outfits = outfits.padded_batch(batch_size, ([None, 299, 299, 3], [None], [None, 1]), drop_remainder=True)

    return outfits\
        .map(append_targets, tf.data.experimental.AUTOTUNE)\
        .prefetch(tf.data.experimental.AUTOTUNE)


def parse_fitb_with_features(raw):
    example = tf.io.parse_single_sequence_example(
        raw, sequence_features={
            "input_categories": tf.io.FixedLenSequenceFeature([], tf.int64),
            "inputs": tf.io.FixedLenSequenceFeature(2048, tf.float32),
            "target_categories": tf.io.FixedLenSequenceFeature([], tf.int64),
            "targets": tf.io.FixedLenSequenceFeature(2048, tf.float32)
        }, context_features={
            "target_position": tf.io.FixedLenFeature([], dtype=tf.int64)
        })
    return example[1]["inputs"], example[1]["input_categories"], \
           example[1]["targets"], example[1]["target_categories"], \
           example[0]["target_position"]


def parse_fitb_with_images(raw):
    example = tf.io.parse_single_sequence_example(
        raw, sequence_features={
            "input_categories": tf.io.FixedLenSequenceFeature([], tf.int64),
            "inputs": tf.io.FixedLenSequenceFeature([], tf.string),
            "target_categories": tf.io.FixedLenSequenceFeature([], tf.int64),
            "targets": tf.io.FixedLenSequenceFeature([], tf.string)
        }, context_features={
            "target_position": tf.io.FixedLenFeature([], dtype=tf.int64)
        })
    return example[1]["inputs"], example[1]["input_categories"], \
           example[1]["targets"], example[1]["target_categories"], \
           example[0]["target_position"]


def get_fitb(filenames, with_features):
    raw_dataset = tf.data.TFRecordDataset(filenames)
    if with_features:
        return raw_dataset.map(parse_fitb_with_features)
    else:
        return raw_dataset.map(parse_fitb_with_images)