from pathlib import Path
import tensorflow as tf
import numpy as np
import json
import argparse

def main():
    """
    Extract features from outfit images and save to .tfrecord files
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, help="Path to dataset root directory", required=True)
    parser.add_argument("--dataset-file", type=str, help="Path to dataset .json file", required=True)
    parser.add_argument("--tfrecord-template", type=str, help="Template for .tfrecord file names", required=True)
    parser.add_argument("--shard-count", type=int, help="Number of .tfrecord files", required=True)

    args = parser.parse_args()

    dataset_root = args.dataset_root
    dataset_filename = args.dataset_file
    output_template = args.tfrecord_template
    shard_count = args.shard_count

    examples = process_dataset(dataset_root, dataset_filename, with_features=True)

    files_per_dataset = shard_count
    outfits_per_file = len(examples) // files_per_dataset

    if len(examples) % files_per_dataset > 0:
        outfits_per_file = outfits_per_file + 1

    writer = None

    for i in range(len(examples)):
        if i % outfits_per_file == 0:
            if i != 0:
                if writer is not None:
                    writer.close()
            writer = tf.io.TFRecordWriter(
                output_template.format(
                    i // outfits_per_file, files_per_dataset))
        writer.write(examples[i])
        if i == len(examples):
            if writer is not None:
                writer.close()


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def extract_features(model: tf.keras.Model, path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(299, 299))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    return np.reshape(model.predict(img_array), 2048)


def process_dataset(dataset_root, dataset_filename, with_features: bool = False):

    with open(Path(dataset_root, dataset_filename)) as json_file:
        raw_json = json.load(json_file)

        examples = []
        model = tf.keras.applications.inception_v3.InceptionV3(weights="imagenet", include_top=False, pooling="avg")
        for outfit in raw_json:
            set_id = int(outfit["set_id"])
            images = []
            categories = []

            for item in outfit["items"]:
                image_path = Path(dataset_root, "images", str(set_id), str(item["index"]) + ".jpg")
                if with_features:
                    features = extract_features(model, image_path)
                    images.append(features)
                else:
                    with open(image_path, "rb") as img_file:
                        raw_image = img_file.read()
                    images.append(raw_image)

                categories.append(item["categoryid"])

            if with_features:
                outfit_features = {
                    "categories": tf.train.FeatureList(feature=[_int64_feature(f) for f in categories]),
                    "features": tf.train.FeatureList(
                        feature=[tf.train.Feature(float_list=tf.train.FloatList(value=f)) for f in images])
                }
            else:
                outfit_features = {
                    "categories": tf.train.FeatureList(feature=[_int64_feature(f) for f in categories]),
                    "images": tf.train.FeatureList(feature=[_bytes_feature(f) for f in images])
                }

            feature_lists = tf.train.FeatureLists(feature_list=outfit_features)

            example = tf.train.SequenceExample(feature_lists=feature_lists)
            examples.append(example.SerializeToString())

    return examples


main()

