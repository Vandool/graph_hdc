import logging
import optuna
import torch

# 1) Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def objective(trial):
    # log the trial start
    logger.info(f"Starting trial #{trial.number}")

    # suggest hyperparameter
    x = trial.suggest_float("x", -10, 10)
    logger.info(f"  Suggested x = {x:.4f}")

    # do GPU work
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"  Using device: {device}")

    t = torch.tensor(x, device=device)
    loss = (t - 2) ** 2
    loss_val = loss.item()

    # log the loss
    logger.info(f"  Trial #{trial.number} loss = {loss_val:.6f}")

    return loss_val

if __name__ == "__main__":
    # turn Optuna’s own logger up to INFO
    optuna.logging.set_verbosity(optuna.logging.INFO)

    storage_url = (
        "sqlite:////home/ka/ka_iti/ka_zi9629/optuna_db/optuna.sqlite"
        "?check_same_thread=false"
    )

    study = optuna.create_study(
        storage=storage_url,
        study_name="hello_slurm_study",
        direction="minimize",
        load_if_exists=True
    )
    logger.info("Beginning study.optimize()")
    study.optimize(
        objective,
        n_trials=10,    # per job
        n_jobs=1
    )
    logger.info(f"Finished all trials. Best params: {study.best_params}")

    for t in study.trials:
        logger.info(f"Trial {t.number}: value={t.value}, params={t.params}")