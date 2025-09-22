# src/exp/<study_name>/hpo/study.py
import optuna
from optuna.integration import BoTorchSampler

from src.exp.real_nvp_v2.hpo.space import DIRECTION


def load_study(study_name: str, sqlite_path: str) -> optuna.Study:
    """
    Create or load an Optuna study bound to a local SQLite file.
    """
    return optuna.create_study(
        study_name=study_name,
        direction=DIRECTION,
        storage=f"sqlite:///{sqlite_path}",
        load_if_exists=True,
        sampler=BoTorchSampler(seed=42),
        # Add a pruner here if you run in-process (not needed for offline ask/tell)
        # pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=5),
    )
