apd_mixture_weight=(0.9 0.7 0.5 0.3 0.1 0.0)
verifier_size=("small" "large")

for w in "${apd_mixture_weight[@]}"; do
    for v in "${verifier_size[@]}"; do
        echo "Submitting job for apd_mixture_weight=${w} verifier_size=${v}"
        
        python eval.py --model_alias dream --task gsm8k --alg apd --apd_mixture_weight ${w} --verifier_size ${v}
    done
done