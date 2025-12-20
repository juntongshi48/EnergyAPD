file_to_launch="exp/batch_run_apd.sh"
echo "Starting ${file_to_launch}"

task=gsm8k

max_lookahead=(null)
kv_window=(null)
n_parallel_samples=(1 2 3 4 8)
max_unmask=(32 null)
verifier_size=small

for ml in "${max_lookahead[@]}"; do
    for kw in "${kv_window[@]}"; do
        for npl in "${n_parallel_samples[@]}"; do
            for mu in "${max_unmask[@]}"; do
                echo "Submitting job: task=${task}, kv_window=${kw}, max_lookahead=${ml}, n_parallel_samples=${npl}, max_unmask=${mu}"
                sbatch "${file_to_launch}" launch "${task}" "${kw}" "${ml}" "${verifier_size}" "${npl}" "${mu}"
            done
        done
    done
done
