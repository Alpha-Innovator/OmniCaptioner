

export NCCL_DEBUG=info
#export NCCL_SOCKET_IFNAME=ib0,eth
export NCCL_P2P_DISABLE=0
#export NCCL_IB_DISABLE=0 
export NCCL_IB_DISABLE=mlx5_0


export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=eth0

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export OMP_DYNAMIC=TRUE
export OMP_NUM_THREADS=8 #和cpus-per-task保持一致


MASTER_ADDR=$(sinfo -Nh -n $SLURM_NODELIST | head -n 1 | awk '{print $1}')
# 设置随机 MASTER_PORT
while true; do
    MASTER_PORT=$((RANDOM % 10000 + 10000))
    if ! ss -tuln | grep -q ":$MASTER_PORT "; then
        break
    fi
done
export MASTER_PORT
echo "Selected MASTER_PORT: $MASTER_PORT"
export MASTER_ADDR=$MASTER_ADDR
echo $MASTER_ADDR
echo $MASTER_PORT
function makehostfile() {
perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
$slots=8 if $slots==0; # workaround 8 gpu machines
@nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
print map { "$b$_ slots=$slots\n" } @nodes'
}
makehostfile > hostfile

GPUS_PER_NODE=8
NNODES=4
GPUS=$(($GPUS_PER_NODE * $NNODES))
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / WORLD_SIZE))

# Create output directory if it doesn't exist
OUTPUT_DIR='./output/output_omnicap/'
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

MODEL_NAME="Qwen2-VL-7B-Instruct"

echo "Running torchrun with MASTER_ADDR=$MASTER_ADDR and MASTER_PORT=$MASTER_PORT"
#

export MASTER_ADDR=$(sinfo -Nh -n $SLURM_NODELIST | head -n 1 | awk '{print $1}')
OMP_NUM_THREADS=${GPUS} srun torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$GPUS_PER_NODE \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  --max-restarts=3 \
    ./src/training/train_caption.py \
    --deepspeed ./scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --freeze_vision_tower False \
    --tune_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACC \
    --min_pixels $((4 * 28 * 28)) \
    --max_pixels $((6400 * 28 * 28)) \
    --learning_rate 1e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 1 \
    --dataloader_num_workers 4 \
    --group_by_length True \
    --max_length 7000 \
    --meta_path ./data/omni_caption_pretrain.yaml \
2>&1 | tee -a "training_log.txt"
