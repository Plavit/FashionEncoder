import datetime

import kerastuner as kt
import src.models.encoder.fashion_encoder as fashion_enc
import src.models.encoder.params as model_params
import tensorflow as tf
from src.models.encoder.encoder_main import EncoderTask


class FashionModelTuner(kt.Tuner):
    def run_trial(self, trial, **kwargs):
        model = self.hypermodel.build(trial.hyperparameters)  # type: tf.keras.Model
        params = model.get_layer("fashion_encoder").params

        if "checkpoint_dir" in params:
            del params["checkpoint_dir"]  # TODO: Do not set parameters during training

        task = EncoderTask(params)
        print(params, flush=True)
        callbacks = [lambda curr_model, acc, epoch: self.on_epoch_end(trial, curr_model, epoch, logs={'acc': acc})]
        task.train(callbacks)


def build(hp: kt.HyperParameters):
    params = model_params.BASE

    params["learning_rate"] = hp.Choice("learning_rate", [0.005, 0.001, 0.0005, 0.0001], default=0.0005)
    params["hidden_size"] = hp.Choice("hidden_size", [32, 64, 128, 256], default=128)
    params["num_hidden_layers"] = hp.Int("num_hidden_layers", 1, 4, 1, default=1)
    params["num_heads"] = hp.Choice("num_heads", [1, 2, 4, 8, 16, 32], default=16)
    params["filter_size"] = hp.Int("filter_size", 32, 512, 64, default=128)
    params["batch_size"] = hp.Int("batch_size", 32, 256, 1, "log", default=128)
    params["category_merge"] = hp.Choice("category_merge", ["add", "multiply"], default="add")
    params["loss"] = hp.Choice("metric", ["cross", "distance"], default="cross")
    params["margin"] = hp.Choice("margin", [0.1, 0.3, 0.8, 1.0, 2.0, 5.0, 10.0], default=1)

    params["category_dim"] = params["hidden_size"]
    params["mode"] = "train"

    params["dataset_files"] = "/mnt/0/projects/outfit-generation/data/processed/tfrecords/pod-train-000-1.tfrecord"
    params["fitb_file"] = "/mnt/0/projects/outfit-generation/data/processed/tfrecords/pod-fitb-features-valid.tfrecord"
    params["category_file"] = "/mnt/0/projects/outfit-generation/data/raw/polyvore_outfits/categories.csv"

    model = fashion_enc.create_model(params, True)

    return model


def main():
    hp = kt.HyperParameters()
    # hp.Choice("margin", [0.1, 0.3, 0.8, 1.0, 2.0, 5.0, 10.0], default=1)
    # hp.Choice("batch_size", [16, 32, 64, 128, 256], default=128)
    hp.Choice("hidden_size", [32, 64, 128, 256], default=128)
    hp.Int("num_hidden_layers", 1, 3, 1, default=1)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tuner = FashionModelTuner(
        oracle=kt.oracles.BayesianOptimization(
            objective=kt.Objective("acc", "max"),
            max_trials=30,
            tune_new_entries=False,
            hyperparameters=hp
        ),
        hypermodel=build,
        project_name="fashion_encoder_training_" + current_time,
        directory="tuner_results"
    )

    tuner.search_space_summary()

    tuner.search()

    best_hps = tuner.get_best_hyperparameters()[0]
    print(best_hps.values, flush=True)


if __name__ == "__main__":
    main()
