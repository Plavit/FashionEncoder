import argparse
import datetime
import time

import tensorflow as tf
import src.models.transformer.metrics as metrics
import src.models.transformer.fashion_encoder as fashion_enc
import src.data.input_pipeline as input_pipeline


class EncoderTask:

    def __init__(self, params):
        self.params = params
        self.model = fashion_enc.create_model(params, True)

    @staticmethod
    def _grad(model: tf.keras.Model, inputs, targets, num_replicas=1):
        with tf.GradientTape() as tape:
            loss_value = metrics.xentropy_loss(model(inputs, training=True), targets) / num_replicas
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def train(self):

        print(self.params, flush=True)

        train_dataset = input_pipeline.get_training_dataset(self.params["dataset_files"], self.params["batch_size"])
        num_epochs = self.params["epoch_count"]
        optimizer = tf.optimizers.Adam()

        model = fashion_enc.create_model(self.params, True)
        model.summary()

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        batch_number = 1

        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)

        if "checkpoint_dir" in self.params:
            manager = tf.train.CheckpointManager(ckpt, self.params["checkpoint_dir"], max_to_keep=3)
        else:
            manager = tf.train.CheckpointManager(ckpt, "./logs/" + current_time + "/tf_ckpts", max_to_keep=3)

        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint), flush=True)
        else:
            print("Initializing from scratch.", flush=True)

        for epoch in range(num_epochs):
            epoch_loss_avg = tf.keras.metrics.Mean('epoch_loss')
            train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
            categorical_acc = tf.metrics.CategoricalAccuracy()
            # Training loop
            for x, y in train_dataset:
                # Optimize the model
                loss_value, grads = self._grad(model, x, y)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                ckpt.step.assign_add(1)

                # Track progress
                epoch_loss_avg(loss_value)  # Add current batch loss
                train_loss(loss_value)
                metrics.categorical_acc(x, y, categorical_acc)

                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss.result(), step=batch_number)
                batch_number = batch_number + 1
            with train_summary_writer.as_default():
                tf.summary.scalar('epoch_loss', epoch_loss_avg.result(), step=epoch)
                tf.summary.scalar('epoch_acc', categorical_acc.result(), step=epoch)

            if epoch % 1 == 0:
                print("Epoch {:03d}: Loss: {:.3f}, Acc: {:.3f}".format(epoch, epoch_loss_avg.result(), categorical_acc.result()))
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

        save_path = manager.save()
        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        print("Trained on " + str(batch_number) + " batches in total.")

    def train_multi(self):

        print(self.params, flush=True)

        strategy = tf.distribute.MirroredStrategy()

        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        train_dataset = input_pipeline.get_training_dataset(self.params["dataset_files"], self.params["batch_size"])
        train_dataset = strategy.experimental_distribute_dataset(train_dataset)
        num_epochs = self.params["epoch_count"]

        with strategy.scope():
            optimizer = tf.optimizers.Adam()
            model = fashion_enc.create_model(self.params, True)
            model.summary()

            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = 'logs/' + current_time + '/train'
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)

            @tf.function
            def train_step(x, y):
                loss_value, grads = self._grad(model, x, y, strategy.num_replicas_in_sync)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                ckpt.step.assign_add(1)
                # Track progress
                epoch_loss_avg(loss_value)  # Add current batch loss
                train_loss(loss_value)

            @tf.function
            def distributed_train_step(x, y):
                per_replica_losses = strategy.experimental_run_v2(train_step, args=(x, y))
                return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

            batch_number = 1
            ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)

            if "checkpoint_dir" in self.params:
                manager = tf.train.CheckpointManager(ckpt, self.params["checkpoint_dir"], max_to_keep=3)
            else:
                manager = tf.train.CheckpointManager(ckpt, "./logs/" + current_time + "/tf_ckpts", max_to_keep=3)

            if manager.latest_checkpoint:
                print("Restored from {}".format(manager.latest_checkpoint), flush=True)
            else:
                print("Initializing from scratch.", flush=True)

            for epoch in range(num_epochs):
                epoch_loss_avg = tf.keras.metrics.Mean('epoch_loss')
                train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
                # Training loop
                for x, y in train_dataset:
                    distributed_train_step(x, y)
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', train_loss.result(), step=batch_number)
                    batch_number = batch_number + 1

                with train_summary_writer.as_default():
                    tf.summary.scalar('epoch_loss', epoch_loss_avg.result(), step=epoch)

                if epoch % 1 == 0:
                    print("Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))
                    save_path = manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            print("Trained on " + str(batch_number) + " batches in total.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-files", type=str, nargs="+", help="Paths to dataset files")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--epoch-count", type=int, help="Number of epochs")
    parser.add_argument("--mode", type=str, help="Type of action", choices=["train", "train_multi"], required=True)
    parser.add_argument("--hidden-size", type=int, help="Hidden size")
    parser.add_argument("--num-heads", type=int, help="Number of heads")
    parser.add_argument("--num-hidden-layers", type=int, help="Number of hidden layers")
    parser.add_argument("--checkpoint-dir", type=str, help="Checkpoint directory")

    args = parser.parse_args()

    arg_dict = vars(args)

    filtered = {k: v for k, v in arg_dict.items() if v is not None}

    if "dataset_files" in arg_dict and not isinstance(arg_dict["dataset_files"], list):
        arg_dict["dataset_files"] = [arg_dict["dataset_files"]]

    params = {
        "feature_dim": 2048,
        "dtype": "float32",
        "hidden_size": 2048,
        "extra_decode_length": 0,
        "num_hidden_layers": 1,
        "num_heads": 2,
        "max_length": 10,
        "default_batch_size": 128,
        "filter_size": 1024,
        "layer_postprocess_dropout": 0.1,
        "attention_dropout": 0.1,
        "relu_dropout": 0.1
    }

    params.update(filtered)

    start_time = time.time()

    task = EncoderTask(params)

    if args.mode == "train":
        task.train()
    elif args.mode == "train_multi":
        task.train_multi()
    else:
        print("Invalid mode")

    end_time = time.time()
    elapsed = end_time - start_time

    print("Task completed in " + str(elapsed) + " seconds")


main()
