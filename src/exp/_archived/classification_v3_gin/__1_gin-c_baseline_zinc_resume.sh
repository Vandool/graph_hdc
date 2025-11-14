JOB_NAME=gin-c_baseline_zinc_resume \
MODEL_NAME=GIN-C \
HV_DIM=7744 \
DATASET=ZINC_SMILES_HRR_7744 \
BATCH_SIZE=256 \
LR=3e-4 \
P_PER_PARENT=20 \
N_PER_PARENT=20 \
EPOCHS=5 \
CONTINUE_FROM=/home/ka/ka_iti/ka_zi9629/projects/graph_hdc/src/exp/classification_v3_gin/results/1_gin/gin-c_baseline_zinc/models/last.ckpt \
./_1_gin_submit.sh
