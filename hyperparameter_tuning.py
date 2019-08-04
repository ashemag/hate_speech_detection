from comet_ml import OfflineExperiment, Optimizer
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

configuration = {
    "algorithm": "bayes",
    "parameters": {
        "dropout": {"type": "float", "min": 0.0, "max": 1.0},
        "num_layers": {"type": "integer", "min": 1, "max": 20},
        "batch_size": {"type": "integer", "min": 64, "max": 2048},
    },
    "spec": {
        "metric": "valid_f_score_hateful",
        "objective": "maximize",
        "seed": 28,
    },
}
optimizer = Optimizer(configuration, api_key=config['DEFAULT']['COMET_API_KEY'], project_name="cnn-phase-4-bert-word")
for experiment in optimizer.get_experiments():
    print(experiment.get_parameter("dropout"))