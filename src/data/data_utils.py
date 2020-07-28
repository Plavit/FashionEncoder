import tensorflow as tf
import numpy as np


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def extract_features(model: tf.keras.Model, path: str) -> np.ndarray:
    """
    Extract features via CNN

    Args:
        model: CNN to use for extraction
        path: Path to img

    Returns: ndarray

    """
    img = tf.keras.preprocessing.image.load_img(path, target_size=(299, 299))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    return np.reshape(model.predict(img_array), 2048)


def key_from_fitb_string(string):
    values = string.split("_")
    return int(values[0]), int(values[1])
