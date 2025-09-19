JOB_NAME=BAH_base_zinc_resume \
PROJ_DIM=1024 \
N_HEADS=6 \
SHARE_PROJ=True \
DROPOUT=0.0 \
NORM=True \
USE_LAYERNORM=True \
USE_TEMPERATURE=True \
HV_DIM=7744 \
DATASET=ZINC_SMILES_HRR_7744 \
VSA=HRR \
BATCH_SIZE=768 \
LR=1.5e-3 \
EPOCHS=5 \
CONTINUE_FROM=/home/hk-project-aimat2/zi9629/graph_hdc/src/exp/classification_v4_mlp_lightning/results/2_bah_lightning/BAH_base_zinc/models/last.ckpt \
./_2_bah_lightning.sh
