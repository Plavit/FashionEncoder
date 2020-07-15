import argparse
import datetime
import logging
import time
from pathlib import Path

import tensorflow as tf
import src.models.encoder.metrics as metrics
import src.models.encoder.fashion_encoder as fashion_enc
import src.data.input_pipeline as input_pipeline
import src.models.encoder.utils as utils
from src.models.encoder import params as params_sets


PARAMS_MAP = {
    "MP": params_sets.MP,
    "MP_ADD": params_sets.MP_ADD,
    "MP_MUL": params_sets.MP,
    "MP_CONCAT": params_sets.MP,
}


class EncoderTask:

    def __init__(self, params):
        self.params = params

    def fitb(self, model, dataset):
        if self.params["loss"] == "cross":
            acc = tf.metrics.CategoricalAccuracy()
        elif self.params["loss"] == "distance":
            acc = tf.metrics.Accuracy()
        else:
            raise RuntimeError("Unexpected loss function")

        mask = tf.constant([[[0]]], dtype=tf.int32)  # FITB mask token is placed at 0th index
        preprocessor = model.get_layer("preprocessor")

        for task in dataset:
            self.fitb_step(model, preprocessor, task, mask, acc)
        return acc.result()

    def fitb_step(self, model, preprocessor, task, mask, acc=None):
        logger = tf.get_logger()
        inputs, input_categories, targets, target_categories, target_position = task
        logger.debug("Targets")
        logger.debug(targets)
        _, targets = preprocessor([targets, target_categories, mask], training=False)
        res = model([inputs, input_categories, mask], training=False)
        outputs = res[0]

        logger.debug("Processed targets")
        logger.debug(targets)
        logger.debug("Outputs")
        logger.debug(outputs)

        debug = self.params["mode"] == "debug"

        if self.params["loss"] == "cross":
            metrics.fitb_acc(outputs, targets, mask, target_position, input_categories, acc, debug)
        elif self.params["loss"] == "distance":
            metrics.outfit_distance_fitb(outputs, targets, mask, target_position, input_categories, acc, debug)
        else:
            raise RuntimeError("Unexpected loss function")

    def debug(self):
        # Create the model
        tf.config.experimental_run_functions_eagerly(True)
        model = fashion_enc.create_model(self.params, is_train=False)
        model.summary()

        if "checkpoint_dir" in self.params:
            ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=model)
            manager = tf.train.CheckpointManager(ckpt, self.params["checkpoint_dir"], max_to_keep=3)
            ckpt.restore(manager.latest_checkpoint)
            if manager.latest_checkpoint:
                print("Restored from {}".format(manager.latest_checkpoint), flush=True)
            else:
                print("Initializing from scratch.", flush=True)

        logger = tf.get_logger()
        logger.setLevel(logging.DEBUG)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + current_time + '/debug'
        debug_summary_writer = tf.summary.create_file_writer(train_log_dir)

        train_dataset, valid_dataset, fitb_dataset = self.get_datasets()
        train_dataset = train_dataset.take(1)
        fitb_dataset = fitb_dataset.take(1)

        logger.debug("-------------- FITB TRACE --------------")
        mask = tf.constant([[[0]]])
        preprocessor = model.get_layer("preprocessor")
        task = tf.data.experimental.get_single_element(fitb_dataset)

        if self.params["loss"] == "cross":
            acc = tf.metrics.CategoricalAccuracy()
        elif self.params["loss"] == "distance":
            acc = tf.metrics.Accuracy()
        else:
            raise RuntimeError("Unexpected loss function")

        tf.summary.trace_on(graph=True)
        self.fitb_step(model, preprocessor, task, mask, acc)
        with debug_summary_writer.as_default():
            tf.summary.trace_export(
                name="fitb_trace",
                step=0)

        logger.debug("-------------- TRAIN TRACE --------------")
        inputs, targets = tf.data.experimental.get_single_element(train_dataset)
        tf.summary.trace_on(graph=True)
        ret = EncoderTask.train_step(model, inputs[0], inputs[1], inputs[2])
        with debug_summary_writer.as_default():
            tf.summary.trace_export(
                name="train_trace",
                step=0)

        logger.debug("-------------- TRAIN METRICS TRACE --------------")
        outputs = ret[0]
        targets = ret[1]
        tf.summary.trace_on(graph=True)

        if self.params["loss"] == "cross":
            acc = tf.metrics.CategoricalAccuracy()
            metrics.xentropy_loss(outputs, tf.stop_gradient(targets), inputs[1], inputs[2],
                                  categorywise_only=self.params["categorywise_train"], debug=True,acc=acc)
        elif self.params["loss"] == "distance":
            acc = tf.metrics.Accuracy()
            metrics.outfit_distance_loss(
                outputs, tf.stop_gradient(targets), inputs[1], inputs[2], self.params["margin"], debug=True, acc=acc)
        else:
            raise RuntimeError("Unexpected loss function")

        with debug_summary_writer.as_default():
            tf.summary.trace_export(
                name="train_metrics_trace",
                step=0)

    @staticmethod
    def train_step(model, inputs, input_categories, mask_positions):
        ret = model([inputs, input_categories, mask_positions], training=True)
        return ret[0], ret[1]

    def _grad(self, model: tf.keras.Model, inputs, targets, acc=None, num_replicas=1, stop_targets_gradient=True):
        with tf.GradientTape() as tape:
            ret = EncoderTask.train_step(model, inputs[0], inputs[1], inputs[2])
            outputs = ret[0]
            targets = tf.stop_gradient(targets) if stop_targets_gradient else ret[1]
            if self.params["loss"] == "cross":
                loss_value = metrics.xentropy_loss(
                    outputs, tf.stop_gradient(targets),
                    inputs[1], inputs[2], acc, categorywise_only=self.params["categorywise_train"]) / num_replicas
            elif self.params["loss"] == "distance":
                loss_value = metrics.outfit_distance_loss(
                    outputs, tf.stop_gradient(targets), inputs[1], inputs[2], self.params["margin"], acc) \
                             / num_replicas
            else:
                raise RuntimeError("Unexpected loss function")
            loss_value += tf.add_n(model.losses)
        grad = tape.gradient(loss_value, model.trainable_variables)
        return loss_value, grad

    def get_datasets(self):
        # Optionally build a lookup table for category groups
        lookup = None
        if self.params["with_category_grouping"]:
            if "category_file" in self.params:
                lookup = utils.build_po_category_lookup_table(self.params["category_file"])
            else:
                lookup = utils.build_category_lookup_table()

        train_dataset = input_pipeline.get_training_dataset(self.params["train_files"],
                                                            self.params["batch_size"],
                                                            not self.params["with_cnn"], lookup)

        if self.params["valid_mode"] == "masking":
            valid_dataset = input_pipeline.get_training_dataset(self.params["valid_files"],
                                                                2,
                                                                not self.params["with_cnn"], lookup).cache()
        else:
            valid_dataset = input_pipeline.get_fitb_dataset([self.params["valid_files"]], not self.params["with_cnn"],
                                                        lookup, self.params["use_mask_category"]).batch(1)

        test_dataset = input_pipeline.get_fitb_dataset([self.params["test_files"]], not self.params["with_cnn"],
                                                       lookup, self.params["use_mask_category"]).batch(1)

        return train_dataset, valid_dataset, test_dataset

    def _validate(self, model, valid_dataset):
        # Validation loop
        valid_loss = tf.keras.metrics.Mean('valid_loss', dtype=tf.float32)
        valid_acc = tf.metrics.CategoricalAccuracy()
        for x, y in valid_dataset:
            ret = model([x[0], x[1], x[2]], training=False)
            outputs = ret[0]
            targets = ret[1]
            loss_value = metrics.xentropy_loss(outputs, targets, x[1], x[2], valid_acc)
            # Track
            valid_loss(loss_value)
        return valid_acc.result()

    def train(self, on_epoch_end=None):
        if on_epoch_end is None:
            on_epoch_end = []

        train_dataset, valid_dataset, test_dataset = self.get_datasets()

        num_epochs = self.params["epoch_count"]
        optimizer = tf.optimizers.Adam(self.params["learning_rate"])

        # Prepare logging
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        batch_number = 0
        if "checkpoint_dir" not in self.params:
            self.params["checkpoint_dir"] = "./logs/" + current_time + "/tf_ckpts"

        # Create the model
        model = fashion_enc.create_model(self.params, True)
        model.summary()
        test_model = fashion_enc.create_model(self.params, False)

        # Threshold of valid acc when target gradient is not stopped
        max_valid = 0

        # Set up checkpointing
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
        manager = tf.train.CheckpointManager(ckpt, self.params["checkpoint_dir"], max_to_keep=3)
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint), flush=True)
        else:
            print("Initializing from scratch.", flush=True)

        if "with_weights" in self.params:
            model.get_layer("preprocessor").load_weights(filepath=self.params["with_weights"] + "preprocessor.h5"
                                                         , by_name=True)
            model.get_layer("encoder").load_weights(filepath=self.params["with_weights"] + "encoder.h5"
                                                         , by_name=True)
            print("Restored weights from {}".format(self.params["with_weights"]), flush=True)

        if "early_stop" in self.params and self.params["early_stop"]:
            early_stopping_monitor = utils.EarlyStoppingMonitor(self.params["early_stop_patience"],
                                                                self.params["early_stop_delta"],
                                                                self.params["early_stop_warmup"])

        for epoch in range(1, num_epochs + 1):
            epoch_loss_avg = tf.keras.metrics.Mean('epoch_loss')
            train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
            if self.params["loss"] == "cross":
                acc = tf.metrics.CategoricalAccuracy()
            elif self.params["loss"] == "distance":
                acc = tf.metrics.Accuracy()
            else:
                raise RuntimeError("Unexpected loss function")

            # Training loop
            for x, y in train_dataset:
                batch_number += 1

                # Optimize the model
                if self.params["target_gradient_from"] == -1:
                    loss_value, grads = self._grad(model, x, y, acc, stop_targets_gradient=False)
                elif max_valid < self.params["target_gradient_from"]:
                    loss_value, grads = self._grad(model, x, y, acc)
                else:
                    loss_value, grads = self._grad(model, x, y, acc, stop_targets_gradient=False)

                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                ckpt.step.assign_add(1)

                # Track progress
                epoch_loss_avg(loss_value)  # Add current batch loss
                train_loss(loss_value)

                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss.result(), step=batch_number)
                    tf.summary.scalar('batch_acc', acc.result(), step=batch_number)

            with train_summary_writer.as_default():
                tf.summary.scalar('epoch_loss', epoch_loss_avg.result(), step=epoch)
                tf.summary.scalar('epoch_acc', acc.result(), step=epoch)

            print("Epoch {:03d}: Loss: {:.3f}, Acc: {:.3f}".format(epoch, epoch_loss_avg.result(),
                                                                   acc.result()))

            if epoch % 2 == 0:
                weights = model.get_weights()
                test_model.set_weights(weights)
                if self.params["valid_mode"] == "masking":
                    valid_acc = self._validate(test_model, valid_dataset)
                else:
                    valid_acc = self.fitb(test_model, valid_dataset)
                print("Epoch {:03d}: Valid Acc: {:.3f}".format(epoch, valid_acc), flush=True)

                with train_summary_writer.as_default():
                    tf.summary.scalar('valid_acc', valid_acc, step=epoch)

                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path), flush=True)

                if valid_acc > max_valid:
                    max_valid = valid_acc
                    model.save_weights(str(Path(self.params["checkpoint_dir"], "best_weights.h5")))
                    model.get_layer("preprocessor").\
                        save_weights(str(Path(self.params["checkpoint_dir"], "preprocessor.h5")))
                    model.get_layer("encoder"). \
                        save_weights(str(Path(self.params["checkpoint_dir"], "encoder.h5")))

                if on_epoch_end is not None:
                    for callback in on_epoch_end:
                        callback(model, valid_acc, epoch)

                if "early_stop" in self.params and self.params["early_stop"]:
                    if early_stopping_monitor.should_stop(valid_acc, 2):
                        print("Stopped the training early. Validation accuracy hasn't improved for {} epochs".format(
                            self.params["early_stop_patience"]))
                        break

        # Test the best model
        test_model.load_weights(str(Path(self.params["checkpoint_dir"], "best_weights.h5")))
        test_acc = self.fitb(test_model, test_dataset)
        print("Test FITB Acc: {:.3f}".format(test_acc), flush=True)

        save_path = manager.save()
        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        print("Trained on " + str(batch_number) + " batches in total.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-files", type=str, nargs="+", help="Paths to train dataset files")
    parser.add_argument("--valid-files", type=str, help="Paths to validation dataset files")
    parser.add_argument("--test-files", type=str, help="Paths to test dataset files")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--filter-size", type=int, help="Transformer filter size")
    parser.add_argument("--epoch-count", type=int, help="Number of epochs")
    parser.add_argument("--mode", type=str, help="Type of action", choices=["train", "debug"], required=True)
    parser.add_argument("--hidden-size", type=int, help="Hidden size")
    parser.add_argument("--num-heads", type=int, help="Number of heads")
    parser.add_argument("--num-hidden-layers", type=int, help="Number of hidden layers")
    parser.add_argument("--checkpoint-dir", type=str, help="Checkpoint directory")
    parser.add_argument("--with-weights", type=str, help="Path to the directory with saved weights")
    parser.add_argument("--masking-mode", type=str, help="Mode of sequence masking",
                        choices=["single-token", "category-masking"])
    parser.add_argument("--valid-mode", type=str, help="Validation mode",
                        choices=["fitb", "masking"])
    parser.add_argument("--learning-rate", type=float, help="Optimizer's learning rate")
    parser.add_argument("--valid-batch-size", type=int,
                        help="Batch size of validation dataset (by default the same as batch size)")
    parser.add_argument("--with-cnn", help="Use CNN to extract features from images", type=utils.str_to_bool, nargs='?',
                        const=True)
    parser.add_argument("--category-embedding", help="Add learned category embedding to image feature vectors",
                        type=utils.str_to_bool, nargs='?', const=True)
    parser.add_argument("--categories-count", type=int, help="Add learned category embedding to image feature vectors")
    parser.add_argument("--with-mask-category-embedding", help="Add category embedding to mask token",
                        type=utils.str_to_bool, nargs='?', const=True)
    parser.add_argument("--target-gradient-from", type=int,
                        help="Value of valid accuracy, when gradient is let through target tensors, -1 for stopped "
                             "gradient")
    parser.add_argument("--info", type=str, help="Additional information about the configuration")
    parser.add_argument("--with-category-grouping", help="Categories are mapped into groups",
                        type=utils.str_to_bool, nargs='?', const=True)
    parser.add_argument("--category-dim", type=int, help="Dimension of category embedding")
    parser.add_argument("--category-merge", type=str, help="Mode of category embedding merge with visual features",
                        choices=["add", "multiply", "concat"])
    parser.add_argument("--use-mask-category", help="Use true masked item category in FITB task",
                        type=utils.str_to_bool, nargs='?', const=True)
    parser.add_argument("--category-file", type=str, help="Path to polyvore outfits categories")
    parser.add_argument("--categorywise-train", help="Compute loss function only between items from the same category",
                        type=utils.str_to_bool, nargs='?', const=True)
    parser.add_argument("--early-stop-patience", type=int, help="Number of epochs to wait for improvement")
    parser.add_argument("--early-stop-delta", type=float, help="Minimum change to qualify as improvement")
    parser.add_argument("--early-stop", help="Enable early stopping",
                        type=utils.str_to_bool)
    parser.add_argument("--loss", type=str, help="Loss function", choices=["cross", "distance"])
    parser.add_argument("--margin", type=float, help="Margin of distance loss function")
    parser.add_argument("--param-set", type=str, help="Name of the hyperparameter set to use as base", default="BASE")

    args = parser.parse_args()

    arg_dict = vars(args)

    filtered = {k: v for k, v in arg_dict.items() if v is not None}

    if "train_files" in arg_dict and not isinstance(arg_dict["train_files"], list):
        arg_dict["train_files"] = [arg_dict["train_files"]]

    if "valid_files" in arg_dict and not isinstance(arg_dict["valid_files"], list):
        arg_dict["valid_files"] = [arg_dict["valid_files"]]

    if "test_files" in arg_dict and not isinstance(arg_dict["test_files"], list):
        arg_dict["test_files"] = [arg_dict["test_files"]]

    params = PARAMS_MAP[arg_dict["param_set"]]

    params.update(filtered)

    print(params, flush=True)

    start_time = time.time()

    task = EncoderTask(params)

    if args.mode == "train":
        task.train()
    elif args.mode == "debug":
        task.debug()
    else:
        print("Invalid mode")

    end_time = time.time()
    elapsed = end_time - start_time

    print("Task completed in " + str(elapsed) + " seconds")


if __name__ == "__main__":
    main()
