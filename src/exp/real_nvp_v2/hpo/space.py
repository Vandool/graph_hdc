import optuna

SPACE = {
    "batch_size": optuna.distributions.IntDistribution(32, 512, log=True),
    "lr": optuna.distributions.FloatDistribution(5e-5, 1e-3, log=True),
    "weight_decay": optuna.distributions.CategoricalDistribution(
        [0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 5e-4]
    ),
    "num_flows": optuna.distributions.IntDistribution(4, 16),
    "num_hidden_channels": optuna.distributions.IntDistribution(256, 2048, step=64),
    "smax_initial": optuna.distributions.FloatDistribution(0.1, 3.0),
    "smax_final": optuna.distributions.FloatDistribution(3.0, 8.0),
    "smax_warmup_epochs": optuna.distributions.IntDistribution(10, 20),
}
DIRECTION = "minimize"
