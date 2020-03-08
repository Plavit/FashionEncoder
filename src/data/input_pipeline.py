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
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    # img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [299, 299])


def parse_example_with_images(raw):
    example = tf.io.parse_single_sequence_example(
        raw, sequence_features={
            "categories": tf.io.FixedLenSequenceFeature([], tf.int64),
            "images": tf.io.FixedLenSequenceFeature([], tf.string)
        })

    decoded_images = []
    raw_imgs = tf.unstack(example[1]["images"])
    for raw_img in raw_imgs:
        img_tensor = decode_img(raw_img)
        decoded_images.append(tf.keras.applications.inception_v3.preprocess_input(img_tensor))

    return tf.stack(decoded_images), example[1]["categories"]


def get_dataset(filenames, with_features):
    raw_dataset = tf.data.TFRecordDataset(filenames)
    if with_features:
        return raw_dataset.map(parse_example_with_features)
    else:
        return raw_dataset.map(parse_example_with_images)


def append_targets(features, categories, token_pos):
    return (features, categories, token_pos), features


def add_special_token_positions(features, categories):
    seq_length = tf.shape(categories)[0]
    random_position = tf.random.uniform((1,), minval=0, maxval=seq_length, dtype="int32")
    token_positions = tf.expand_dims(random_position, 0)
    return features, categories, token_positions


def get_training_dataset(filenames, batch_size, with_features):
    outfits = get_dataset(filenames, with_features)
    outfits = outfits.map(add_special_token_positions, tf.data.experimental.AUTOTUNE)
    outfits = outfits.padded_batch(batch_size, ([None, 2048], [None], [None, 1]), drop_remainder=True)
    return outfits.map(append_targets, tf.data.experimental.AUTOTUNE)\
        .cache()\
        .prefetch(tf.data.experimental.AUTOTUNE)
