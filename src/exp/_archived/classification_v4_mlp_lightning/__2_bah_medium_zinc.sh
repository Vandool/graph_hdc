# ---- BAH: MEDIUM (adds 2-layer projection; moderate dropout) ----
JOB_NAME=BAH_med_zinc \
PROJ_DIM=1536 \
N_HEADS=8 \
PROJ_hidden=3072 \
SHARE_PROJ=False \
DROPOUT=0.10 \
NORM=True \
USE_LAYERNORM=True \
USE_TEMPERATURE=True \
HV_DIM=7744 \
DATASET=ZINC_SMILES_HRR_7744 \
VSA=HRR \
BATCH_SIZE=512 \
LR=1.0e-3 \
EPOCHS=20 \
./_2_bah_lightning.sh
