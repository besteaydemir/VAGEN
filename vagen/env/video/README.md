- Add the video path (or the point cloud path) to the `env_config.py`



- Run with 

python -m vagen.env.video.run_test_env


from the VAGEN folder.

VAGEN/env_config.yaml


DATE=$(date +%y-%m-%d)
TIME=$(date +%H-%M)
LOG_DIR="logs/$DATE/$TIME"
mkdir -p $LOG_DIR

export HYDRA_FULL_ERROR=1

SERVER_LOG="$LOG_DIR/server.log"

# Start server in background
python -m vagen.server.server server.port=5000 use_state_reward=False &> $SERVER_LOG &
SERVER_PID=$!
echo "Server started with PID $SERVER_PID"
ps -ef | grep "[v]agen.server.server"




python -m vagen.env.create_dataset --yaml_path "$PWD/env_config.yaml" --train_path "data/$EXPERIMENT_NAME/train.parquet" --test_path "data/$EXPERIMENT_NAME/test.parquet" 



pip install -e . --no-deps 

nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0
(vagen) di38riq@mcml-hgx-a100-002:~/VAGEN/verl$ gcc --version
gcc (Ubuntu 11.4.0-1ubuntu1~22.04.2) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.



sbatch scripts/examples/room_example/run_tmux.s