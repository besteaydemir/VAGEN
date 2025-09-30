#!/bin/bash
#SBATCH --job-name=vagen_ppo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2              # 1 GPU for server, 1 for training
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00           # adjust as needed
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# -----------------------------
# 1. Environment setup
# -----------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vagen

cd ~/VAGEN

# -----------------------------
# 2. Logging directories
# -----------------------------
# --- Setup logging directories ---
DATE=$(date +%y-%m-%d)
TIME=$(date +%H-%M)
LOG_DIR="logs/$DATE/$TIME"
mkdir -p $LOG_DIR
SERVER_LOG="$LOG_DIR/server.log"
TRAIN_LOG="$LOG_DIR/train.log"

# --- Detect node IP ---
NODE_IP=$(hostname -I | awk '{print $1}')
echo "Running on node $NODE_IP"

# Allocate the GPUs for Ray
export CUDA_VISIBLE_DEVICES=0,1

# Start Ray head (GCS + Redis) on port 6379
ray start --head --port=6379 --dashboard-port=8265 --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=2

NODE_IP=$(hostname -i)
export RAY_ADDRESS="$NODE_IP:6379"


# --- Start the server on GPU 0 ---
CUDA_VISIBLE_DEVICES=0 python -m vagen.server.server \
    server.port=5000 use_state_reward=False \
    &> $SERVER_LOG &
SERVER_PID=$!
echo "Server started with PID $SERVER_PID"

# --- Wait a few seconds for server to come up ---
sleep 10

# --- Run PPO training on GPU 1 ---
CUDA_VISIBLE_DEVICES=1 python -m vagen.trainer.main_ppo \
    algorithm.adv_estimator=masked_gae \
    algorithm.high_level_gamma=0.95 \
    data.train_files=data/$EXPERIMENT_NAME/train.parquet \
    data.val_files=data/$EXPERIMENT_NAME/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=200 \
    data.max_trajectory_length=2400 \
    data.image_key=images \
    data.truncation=left \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.temperature=0.7 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='vagen_new' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=150 \
    trainer.test_freq=20 \
    trainer.total_training_steps=300 \
    rollout_manager.max_turns=3 \
    rollout_manager.window_size=5 \
    rollout_manager.use_multi_turn_reward=False \
    rollout_manager.use_loss_mask=True \
    rollout_manager.use_gae_mask=True \
    trainer.val_before_train=True \
    trainer.val_generations_to_log_to_wandb=8 \
    rollout_manager.n_trajectory=1 \
    rollout_manager.use_service=True \
    rollout_manager.timeout=300 \
    rollout_manager.base_url="http://127.0.0.1:5000" \
    2>&1 | tee $TRAIN_LOG

# --- Wait for server to finish ---
wait $SERVER_PID