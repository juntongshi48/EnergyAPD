debug=$1

model_alias=dream
task=gsm8k
alg=apd
apd_mixture_weight=0.5
max_lookahead=100
verifier_size="large"
qwen_7b_ckpt="Qwen/Qwen2.5-Math-7B-Instruct"

n_parallel_samples=5

output_dir="results"
tag="math-instruct_greedy"

PYTHON_PREFIX=""
limit=null
if [ "$debug" = "debug" ]; then
    PYTHON_PREFIX="python -m debugpy --listen 0.0.0.0:5678 --wait-for-client"
    output_dir="results_debug"
    limit=2
    echo "In DEBUG mode"
else
    PYTHON_PREFIX="python"
fi

gpu_id=0
CUDA_VISIBLE_DEVICES=${gpu_id} \
${PYTHON_PREFIX} \
    eval_mp.py \
    output_dir=${output_dir} \
    model_alias=${model_alias} \
    task=${task} \
    alg=${alg} \
    limit=${limit} \
    apd_mixture_weight=${apd_mixture_weight} \
    n_parallel_samples=${n_parallel_samples}
    max_lookahead=${max_lookahead} \
    verifier_size=${verifier_size} \
    tag=${tag} \
    qwen_7b_ckpt=${qwen_7b_ckpt}

# qwen_7b_ckpt=${qwen_7b_ckpt} \
# max_lookahead=${max_lookahead} \
