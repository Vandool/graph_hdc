# ---- BAH: LARGE (more heads & width; stronger reg; class-weighting) ----
JOB_NAME=BAH_large_qm9 \
PROJ_DIM=1536 \
N_HEADS=12 \
PROJ_hidden=3072 \
SHARE_PROJ=False \
DROPOUT=0.15 \
NORM=True \
USE_LAYERNORM=True \
USE_TEMPERATURE=True \
HV_DIM=1600 \
DATASET=QM9_SMILES_HRR_1600 \
VSA=HRR \
BATCH_SIZE=256 \
LR=6e-4 \
EPOCHS=20 \
./_2_bah_lightning.sh
