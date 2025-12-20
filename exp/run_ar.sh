#!/bin/bash
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --qos=normal
#SBATCH --time=1-00:00:00            # Max time (days-hrs:mins:secs)
#SBATCH --nodes=1                    # Single node
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:a6000ada:1       # Request some number and type of GPUs
#SBATCH --job-name=eval_gsm8k # Job name
#SBATCH --output=sout/eval_gsm8k\_%j.out  # Output file (%j expands to jobID)
# )

source /atlas2/u/minkai/miniconda3/etc/profile.d/conda.sh
conda activate apd

echo "Hostname: $(hostname)"

mode=$1

model_alias=qwen0.5b
task=math
alg=leftright
qwen_small_ckpt="Qwen/Qwen2.5-Math-1.5B-Instruct"
qwen_7b_ckpt="Qwen/Qwen2.5-Math-7B-Instruct"

output_dir="results"
tag="math-1.5-instruct"

CMD_PREFIX=""
limit=null
if [ "$mode" = "debug" ]; then
    CMD_PREFIX="python -m debugpy --listen 0.0.0.0:5678 --wait-for-client"
    output_dir="results_debug"
    limit=2
    echo "In DEBUG mode"
elif [ "$mode" = "launch" ]; then
    CMD_PREFIX="srun python"
    echo "In LAUNCH mode"
else
    CMD_PREFIX="python"
fi

gpu_id=0
CUDA_VISIBLE_DEVICES=${gpu_id} \
${CMD_PREFIX} \
    eval_mp.py \
    output_dir=${output_dir} \
    model_alias=${model_alias} \
    task=${task} \
    alg=${alg} \
    limit=${limit} \
    tag=${tag} \
    qwen_small_ckpt=${qwen_small_ckpt} \
    qwen_7b_ckpt=${qwen_7b_ckpt}

# qwen_7b_ckpt=${qwen_7b_ckpt} \
# max_lookahead=${max_lookahead} \