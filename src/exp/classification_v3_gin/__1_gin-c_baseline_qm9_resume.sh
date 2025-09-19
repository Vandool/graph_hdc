JOB_NAME=gin-c_baseline_qm9_resume \
MODEL_NAME=GIN-C \
HV_DIM=1600 \
DATASET=QM9_SMILES_HRR_1600 \
BATCH_SIZE=256 \
LR=3e-4 \
P_PER_PARENT=20 \
N_PER_PARENT=20 \
EPOCHS=5 \
CONTINUE_FROM=/home/ka/ka_iti/ka_zi9629/projects/graph_hdc/src/exp/classification_v3_gin/results/1_gin/gin-c_baseline_qm9/models/last.ckpt \
./_1_gin_submit.sh
