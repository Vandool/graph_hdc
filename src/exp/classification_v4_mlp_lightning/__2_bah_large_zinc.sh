# ---- BAH: LARGE (more heads & width; stronger reg; class-weighting) ----
JOB_NAME=BAH_large_zinc \
PROJ_DIM=2048 \
N_HEADS=12 \
PROJ_hidden=4096 \
SHARE_PROJ=False \
DROPOUT=0.15 \
NORM=True \
USE_LAYERNORM=True \
USE_TEMPERATURE=True \
HV_DIM=7744 \
DATASET=ZINC_SMILES_HRR_7744 \
VSA=HRR \
BATCH_SIZE=256 \
LR=6e-4 \
EPOCHS=20 \
./_2_bah_lightning.sh
