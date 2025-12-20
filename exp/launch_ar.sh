file_to_launch="exp/batch_run_ar.sh"
echo "Starting ${file_to_launch}"

task=gsm8k

sbatch "${file_to_launch}" launch "${task}"