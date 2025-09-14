# ---- BAH: MEDIUM (adds 2-layer projection; moderate dropout) ----
JOB_NAME=BAH_med \
PROJ_DIM=1024 \
N_HEADS=8 \
PROJ_hidden=2048 \
SHARE_PROJ=False \
DROPOUT=0.10 \
NORM=True \
USE_LAYERNORM=True \
USE_TEMPERATURE=True \
HV_DIM=1600 \
DATASET=QM9_SMILES_HRR_1600 \
VSA=HRR \
BATCH_SIZE=512 \
LR=1.0e-3 \
WEIGHT_DECAY=3e-4 \
EPOCHS=20 \
./_2_bah_lightning.sh
