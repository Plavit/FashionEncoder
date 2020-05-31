import kerastuner as kt
import src.models.encoder.fashion_encoder as fashion_enc
import src.models.encoder.params as model_params
from src.models.encoder.encoder_main import EncoderTask


class FashionModelTuner(kt.Tuner):
    def run_trial(self, trial, **kwargs):
        model = self.hypermodel.build(trial.hyperparameters)  # type: fashion_enc.FashionEncoder
        task = EncoderTask(model.params)
        callbacks = [lambda curr_model, acc, epoch: self.on_epoch_end(trial, curr_model, epoch, logs={'acc': acc})]
        task.train(callbacks)


class HyperFashionEncoder(kt.HyperModel):
    def build(self, hp: kt.HyperParameters):
        params = model_params.BASE

        params["learning_rate"] = hp.Choice("learning_rate", [0.005, 0.001, 0.0005, 0.0001], default=0.0005)
        params["hidden_size"] = hp.Int("hidden_size", 64, 2048, 1, "log", default=1024)
        params["num_hidden_layers"] = hp.Int("num_hidden_layers", 1, 2, 1, default=1)
        params["num_heads"] = hp.Int("num_heads", 1, 32, 1, "log", default=16)
        params["filter_size"] = hp.Int("filter_size", 512, 2048, 256, default=1024)
        params["batch_size"] = hp.Int("batch_size", 32, 256, 1, "log", default=128)
        params["category_merge"] = hp.Choice("category_merge", ["add", "multiply"], default="add")

        params["mode"] = "train"

        model = fashion_enc.create_model(params, True)

        return model


def main():
    tuner = FashionModelTuner(
        oracle=kt.oracles.BayesianOptimization(
            objective=kt.Objective("acc", "max"),
            max_trials=1),
        hypermodel=HyperFashionEncoder,
        project_name="fashion_encoder_training",
        directory="tuner_results"
    )

    tuner.search_space_summary()

    # tuner.search()

    best_hps = tuner.get_best_hyperparameters()[0]
    print(best_hps.values, flush=True)


if __name__ == "__main__":
    main()
