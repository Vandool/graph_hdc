import optuna


def objective(trial):
    # 1) Suggest a float “x” in [-10, 10]
    x = trial.suggest_float("x", -10, 10)
    # 2) Return the value we want to minimize: (x – 2)²
    return (x - 2) ** 2


if __name__ == "__main__":
    # 3) Create a new study (in-memory SQLite by default)
    study = optuna.create_study(direction="minimize")
    # 4) Run 100 trials sequentially
    study.optimize(objective, n_trials=100)
    # 5) Print best parameter found
    print(study.best_params)