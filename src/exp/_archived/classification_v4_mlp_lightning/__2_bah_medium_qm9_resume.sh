# ---- BAH: MEDIUM (adds 2-layer projection; moderate dropout) ----
JOB_NAME=BAH_med_qm9_resume \
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
EPOCHS=20 \
CONTINUE__FROM=/home/iti/zi9629/graph_hdc/src/exp/classification_v4_mlp_lightning/results/2_bah_lightning/BAH_med_qm9/models/last.ckpt \
./_2_bah_lightning.sh
