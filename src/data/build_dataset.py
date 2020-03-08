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
    parser.add_argument("--with-features", help="With CNN features extracted", action='store_true')

    args = parser.parse_args()

    dataset_root = args.dataset_root
    dataset_filename = args.dataset_file
    output_template = args.tfrecord_template
    shard_count = args.shard_count
    with_features = args.with_features

    print("Arguments parsed", flush=True)

    examples = process_dataset(dataset_root, dataset_filename, with_features=with_features)

    print("Processed " + str(len(examples)) + " examples", flush=True)

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
    print("Saved the dataset successfully", flush=True)


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
    excluded_categories = [77, 76, 78, 113, 115, 116, 118, 120, 122, 123, 124, 126, 127, 129, 130, 132, 135, 136, 139, 140, 141, 143, 144, 4241, 4242, 147, 4244, 150, 4247, 4248, 153, 156, 154, 155, 157, 4254, 159, 160, 4257, 162, 163, 164, 166, 167, 168, 169, 170, 4267, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 231, 311, 313, 314, 316, 317, 321, 4432, 4433, 4437, 4439, 4440, 4441, 4442, 4443, 4445, 4446, 4448, 4449, 4450, 4451, 4478, 4480, 4481, 4482, 4483, 4484, 4485, 4486, 4487, 4488, 4489, 4490, 4492, 4493, 4499, 4500, 4501, 4502, 4503, 4504, 4505, 4506, 4507, 4508, 4509, 4510, 4511, 4512, 4513, 4512, 4438, 4240, 146, 148, 4246, 151, 152, 949, 161, 4258, 171, 4269, 4276, 3336, 1967, 4431, 5535, 4436, 4430,
                93, 94, 95, 96, 97, 98, 99, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 4292, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 211, 213, 214, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 319, 320, 333, 334, 335, 338, 339, 340, 196, 336, 337]
    total_disposed = 0
    outfits_disposed = 0
    with open(Path(dataset_root, dataset_filename)) as json_file:
        raw_json = json.load(json_file)
        print("Loaded " + str(len(raw_json)) + " items", flush=True)
        examples = []
        model = tf.keras.applications.inception_v3.InceptionV3(weights="imagenet", include_top=False, pooling="avg")

        for outfit in raw_json:
            set_id = int(outfit["set_id"])
            images = []
            categories = []

            for item in outfit["items"]:
                if item["categoryid"] in excluded_categories:
                    total_disposed = total_disposed + 1
                    continue

                image_path = Path(dataset_root, "images", str(set_id), str(item["index"]) + ".jpg")
                if with_features:
                    features = extract_features(model, image_path)
                    images.append(features)
                else:
                    with open(image_path, "rb") as img_file:
                        raw_image = img_file.read()
                    images.append(raw_image)

                categories.append(item["categoryid"])

            if len(images) < 3:
                outfits_disposed = outfits_disposed + 1
                continue

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

    print("Disposed " + str(total_disposed) + " products")
    print("Disposed " + str(outfits_disposed) + " outfits")
    return examples


if __name__ == "__main__":
    main()
