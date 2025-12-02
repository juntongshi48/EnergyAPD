debug=$1

gpu_id=6

# if [ "$debug" = "true" ]; then
#     CUDA_VISIBLE_DEVICES=${gpu_id} \
#     python -m debugpy --listen 0.0.0.0:5678 --wait-for-client eval.py --model_alias dream --task gsm8k --alg apd --apd_mixture_weight 0.5
# else
#     CUDA_VISIBLE_DEVICES=${gpu_id} \
#     python eval.py --model_alias dream --task gsm8k --alg apd --apd_mixture_weight 0.5
# fi

model_alias=dream
task=gsm8k
alg=apd
apd_mixture_weight=0.0
# max_lookahead=100
verifier_size="large"
qwen_7b_ckpt="Qwen/Qwen2.5-Math-7B-Instruct"

output_dir="results"
tag="math-instruct_k1_greedy"

PYTHON_PREFIX=""

if [ "$debug" = "true" ]; then
    PYTHON_PREFIX="python -m debugpy --listen 0.0.0.0:5678 --wait-for-client"
    output_dir="results_debug"
else
    PYTHON_PREFIX="python"
fi

CUDA_VISIBLE_DEVICES=${gpu_id} \
${PYTHON_PREFIX} \
    eval_mp.py \
    model_alias=${model_alias} \
    task=${task} \
    alg=${alg} \
    apd_mixture_weight=${apd_mixture_weight} \
    verifier_size=${verifier_size} \
    tag=${tag} \
    qwen_7b_ckpt=${qwen_7b_ckpt}

# qwen_7b_ckpt=${qwen_7b_ckpt} \
# max_lookahead=${max_lookahead} \
