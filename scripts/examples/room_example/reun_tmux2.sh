#!/bin/bash
#SBATCH --job-name=vagen_ppo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00
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
DATE=$(date +%y-%m-%d)
TIME=$(date +%H-%M)
LOG_DIR="logs/$DATE/$TIME"
mkdir -p $LOG_DIR
SERVER_LOG="$LOG_DIR/server.log"
TRAIN_LOG="$LOG_DIR/train.log"

# -----------------------------
# 3. Experiment & port setup
# -----------------------------
PORT=5000
SERVER_CUDA=0
TRAIN_CUDA=1
EXPERIMENT_NAME="slurm_vagen_exp"

unset RAY_ADDRESS   # prevent Ray from trying to connect externally

# -----------------------------
# 4. Function to create tmux sessions
# -----------------------------
find_available_session() {
    local base_name=$1
    local count=0
    while tmux has-session -t "${base_name}${count}" 2>/dev/null; do
        count=$((count+1))
    done
    echo "${base_name}${count}"
}

SERVER_SESSION=$(find_available_session "server")
TRAIN_SESSION=$(find_available_session "train")

# -----------------------------
# 5. Start the server in tmux
# -----------------------------
tmux new-session -d -s "$SERVER_SESSION"
tmux send-keys -t "$SERVER_SESSION" "conda activate vagen" C-m
tmux send-keys -t "$SERVER_SESSION" "export CUDA_VISIBLE_DEVICES=$SERVER_CUDA" C-m
tmux send-keys -t "$SERVER_SESSION" "unset RAY_ADDRESS" C-m
tmux send-keys -t "$SERVER_SESSION" "python -m vagen.server.server server.port=$PORT use_state_reward=False &> $SERVER_LOG" C-m

echo "Server tmux session: $SERVER_SESSION"

# Give server time to start
sleep 10

# -----------------------------
# 6. Start the trainer in tmux
# -----------------------------
tmux new-session -d -s "$TRAIN_SESSION"
tmux send-keys -t "$TRAIN_SESSION" "cd ~/VAGEN" C-m
tmux send-keys -t "$TRAIN_SESSION" "conda activate vagen" C-m
tmux send-keys -t "$TRAIN_SESSION" "export CUDA_VISIBLE_DEVICES=$TRAIN_CUDA" C-m
tmux send-keys -t "$TRAIN_SESSION" "unset RAY_ADDRESS" C-m
tmux send-keys -t "$TRAIN_SESSION" "python -m vagen.trainer.main_ppo \
    rollout_manager.base_url=\"http://127.0.0.1:$PORT\" \
    trainer.experiment_name=$EXPERIMENT_NAME &> $TRAIN_LOG" C-m

echo "Trainer tmux session: $TRAIN_SESSION"

# -----------------------------
# 7. Summary
# -----------------------------
echo "Server session: $SERVER_SESSION"
echo "Trainer session: $TRAIN_SESSION"
echo "To attach to server: tmux attach -t $SERVER_SESSION"
echo "To attach to trainer: tmux attach -t $TRAIN_SESSION"