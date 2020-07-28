import argparse
import json
from pathlib import Path
import tensorflow as tf
import src.data.build_dataset as build_dataset
import src.data.data_utils as utils


def main():
    """
        Build a dataset with a FITB task and save it to .tfrecord file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, help="Path to dataset root directory", required=True)
    parser.add_argument("--dataset-file", type=str, help="Filename of dataset .json file", required=True)
    parser.add_argument("--output-path", type=str, help="Path to output file", required=True)
    parser.add_argument("--fitb-file", type=str, help="Filename of FITB .json file", required=True)
    parser.add_argument("--with-features", help="With CNN features extracted", action='store_true')

    args = parser.parse_args()

    dataset_root = args.dataset_root
    dataset_filename = args.dataset_file
    with_features = args.with_features
    output_path = args.output_path
    fitb_filename = args.fitb_file

    examples = build_fitb(dataset_root, dataset_filename, with_features, fitb_filename)
    with tf.io.TFRecordWriter(output_path) as writer:
        for i in range(len(examples)):
            writer.write(examples[i])

    print("Saved the fitb successfully", flush=True)


def build_fitb(dataset_root: str, dataset_filename: str, with_features: bool, fitb_filename: str):
    """
    Create list of tf.SequenceExample that represents the fill-in-the-blank task

    Args:
        dataset_root: Path to the root of the dataset
        dataset_filename: Dataset filename
        with_features: bol whether to use CNN to extract features
        fitb_filename: Filename of the FITB file

    Returns: List of tf.SequenceExample that represent FITB samples

    """
    with open(Path(dataset_root, dataset_filename)) as json_file:
        raw_json = json.load(json_file)
        print("Loaded " + str(len(raw_json)) + " items", flush=True)
        items = {}

        if with_features:
            model = tf.keras.applications.inception_v3.InceptionV3(weights="imagenet", include_top=False, pooling="avg")

        # Load all test items into dict
        for outfit in raw_json:
            set_id = int(outfit["set_id"])

            for item in outfit["items"]:
                image_path = Path(dataset_root, "images", str(set_id), str(item["index"]) + ".jpg")
                if with_features:
                    features = utils.extract_features(model, image_path)
                    items.update({(set_id, item["index"]): (features, item["categoryid"])})
                else:
                    with open(image_path, "rb") as img_file:
                        raw_image = img_file.read()
                    items.update({(set_id, item["index"]): (raw_image, item["categoryid"])})
    examples = []
    with open(Path(dataset_root, fitb_filename)) as fitb_file:
        raw_json = json.load(fitb_file)
        print("Loaded " + str(len(raw_json)) + " questions", flush=True)

        # Compose questions from FITB file and test items dict
        for task in raw_json:
            set_id = None
            inputs = []
            input_categories = []
            targets = []
            target_categories = []
            target_pos = None

            for question_item_str in task["question"]:
                q_key = utils.key_from_fitb_string(question_item_str)
                item_features, item_category = items[q_key]
                inputs.append(item_features)
                input_categories.append(item_category)
                set_id = q_key[0]
            pos = 0

            for question_item_str in task["answers"]:
                q_key = utils.key_from_fitb_string(question_item_str)
                if q_key[0] == set_id:
                    target_pos = pos
                item_features, item_category = items[q_key]
                targets.append(item_features)
                target_categories.append(item_category)
                pos += 1

            if with_features:
                question_features = {
                    "input_categories": tf.train.FeatureList(feature=[utils.int64_feature(f) for f in input_categories]),
                    "inputs": tf.train.FeatureList(
                        feature=[tf.train.Feature(float_list=tf.train.FloatList(value=f)) for f in inputs]),
                    "target_categories": tf.train.FeatureList(feature=[utils.int64_feature(f) for f in target_categories]),
                    "targets": tf.train.FeatureList(
                        feature=[tf.train.Feature(float_list=tf.train.FloatList(value=f)) for f in targets])
                }
            else:
                question_features = {
                    "input_categories": tf.train.FeatureList(feature=[utils.int64_feature(f) for f in input_categories]),
                    "inputs": tf.train.FeatureList(feature=[utils.bytes_feature(f) for f in inputs]),
                    "target_categories": tf.train.FeatureList(feature=[utils.int64_feature(f) for f in target_categories]),
                    "targets": tf.train.FeatureList(feature=[utils.bytes_feature(f) for f in targets])
                }
            feature_lists = tf.train.FeatureLists(feature_list=question_features)
            context = {
                "target_position": utils.int64_feature(target_pos)
            }
            context = tf.train.Features(feature=context)
            example = tf.train.SequenceExample(feature_lists=feature_lists, context=context)
            examples.append(example.SerializeToString())
        return examples


if __name__ == "__main__":
    main()
