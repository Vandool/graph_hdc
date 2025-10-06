# ---- BAH: BASELINE (fast, stable; shared proj, no hidden MLP) ----
JOB_NAME=BAH_base_hp_resample_qm9 \
PROJ_DIM=768 \
N_HEADS=6 \
SHARE_PROJ=True \
DROPOUT=0.0 \
NORM=True \
USE_LAYERNORM=True \
USE_TEMPERATURE=True \
HV_DIM=1600 \
DATASET=QM9_SMILES_HRR_1600 \
VSA=HRR \
BATCH_SIZE=256 \
LR=1.5e-3 \
EPOCHS=30 \
./_2_bah_lightning.sh
