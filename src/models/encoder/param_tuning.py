import datetime

import kerastuner as kt
import src.models.encoder.fashion_encoder as fashion_enc
import src.models.encoder.params as model_params
import tensorflow as tf
from src.models.encoder.encoder_main import EncoderTask


class FashionModelTuner(kt.Tuner):
    def run_trial(self, trial, **kwargs):
        model = self.hypermodel.build(trial.hyperparameters)  # type: tf.keras.Model
        params = model.get_layer("encoder").params

        if "checkpoint_dir" in params:
            del params["checkpoint_dir"]  # Do not use checkpoints

        task = EncoderTask(params)
        print(params, flush=True)
        callbacks = [lambda curr_model, acc, epoch: self.on_epoch_end(trial, curr_model, epoch, logs={'acc': acc})]
        task.train(callbacks)


def build(hp: kt.HyperParameters):
    params = model_params.PO_ADD

    params["learning_rate"] = hp.Choice("learning_rate", [0.005, 0.001, 0.0005, 0.0001], default=0.0005)
    params["batch_size"] = hp.Choice("batch_size", [32, 64, 96, 128], default=128)
    params["hidden_size"] = hp.Choice("hidden_size", [32, 64, 128, 256], default=256)
    params["num_hidden_layers"] = hp.Int("num_hidden_layers", 1, 4, 1, default=2)
    params["num_heads"] = hp.Choice("num_heads", [1, 2, 4, 8, 16, 32], default=32)
    params["filter_size"] = params["hidden_size"] * 2
    params["category_merge"] = hp.Choice("category_merge", ["add", "multiply"], default="add")
    params["loss"] = hp.Choice("metric", ["cross", "distance"], default="cross")
    params["margin"] = hp.Choice("margin", [0.1, 0.3, 0.8, 1.0, 2.0, 5.0, 10.0], default=1)

    params["category_dim"] = params["hidden_size"]
    params["mode"] = "train"

    params["layer_postprocess_dropout"] = hp.Choice("layer_postprocess_dropout", [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                                                    default=0.1)
    params["attention_dropout"] = hp.Choice("attention_dropout", [0.05, 0.1, 0.2, 0.3, 0.4, 0.5], default=0.1)
    params["relu_dropout"] = hp.Choice("relu_dropout", [0.05, 0.1, 0.2, 0.3, 0.4, 0.5], default=0.1)
    params["dense_regularization"] = hp.Choice("dense_regularization", [0.0, 0.0005, 0.001, 0.005, 0.01], default=0)
    params["emb_dropout"] = hp.Choice("emb_dropout", [0.0, 0.05, 0.1, 0.2, 0.3], default=0)
    params["i_dense_dropout"] = hp.Choice("i_dense_dropout", [0.05, 0.1, 0.2, 0.3], default=0.1)

    model = fashion_enc.create_model(params, True)

    return model


def main():
    hp = kt.HyperParameters()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tuner = FashionModelTuner(
        oracle=kt.oracles.BayesianOptimization(
            objective=kt.Objective("acc", "max"),
            max_trials=100,
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
