import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
import argparse
import json
import logging
import torch
from lm_eval import evaluator
from harness import DreamEvalHarness, ProfileEvalHarness, LladaEvalHarness
from utils import parse_results
from transformers import AutoModel, AutoTokenizer
from dream.modeling_dream import DreamModel

import torch.multiprocessing as mp
import hydra


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply custom task configurations
from eval_config.monkey_patch import apply_custom_task_configs
apply_custom_task_configs()

# Define the tasks you want to support
TASKS = {'gsm8k': 'gsm8k', 'math': 'hendrycks_math', 'gpqa': 'gpqa_main_generative_n_shot', 'humaneval': 'humaneval_instruct'} # Use the exact task names from lm-eval task registry


def get_model(args):
    
    model_alias = args.model_alias
    alg = args.alg
    # Canonical names only
    tokens_per_step = args.tokens_per_step
    max_lookahead = args.max_lookahead
    kv_window = args.kv_window
    apd_mixture_weight = args.apd_mixture_weight
    # %%% New Dev Configs %%%
    n_parallel_samples = args.n_parallel_samples
    max_unmask = args.max_unmask
    # %%% New Dev Configs %%%
    num_steps = args.num_steps
    task_name = args.task

    logger.info(f"Configuring model details for alias: {model_alias}")
    if model_alias == "qwen7b":
        if args.qwen_7b_ckpt is None:
            raise ValueError("--qwen_7b_ckpt is required when --model_alias=qwen7b")
        model = ProfileEvalHarness(pretrained=args.qwen_7b_ckpt, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda", max_length=16384)
    elif model_alias == "qwen0.5b":
        if args.qwen_small_ckpt is None:
            raise ValueError("--qwen_small_ckpt is required when --model_alias=qwen0.5b")
        model = ProfileEvalHarness(pretrained=args.qwen_small_ckpt, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda", max_length=16384)
    elif model_alias == "dream":
        if args.dream_ckpt is None:
            raise ValueError("--dream_ckpt is required when --model_alias=dream")
        if args.alg == "apd" and args.qwen_small_ckpt is None:
            raise ValueError("--qwen_small_ckpt is required when --model_alias=dream and --alg=apd")
        dream = DreamModel.from_pretrained(args.dream_ckpt, 
                                               trust_remote_code=True,  
                                               attn_implementation="sdpa", 
                                               torch_dtype=torch.bfloat16, 
                                               device_map="cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.dream_ckpt, trust_remote_code=True)
        verifier_ckpt = None
        if args.verifier_size =='small':
            verifier_ckpt = args.qwen_small_ckpt
        elif args.verifier_size == 'large':
            verifier_ckpt = args.qwen_7b_ckpt
        model = DreamEvalHarness(
            pretrained=dream,
            tokenizer=tokenizer,
            alg=alg,
            tokens_per_step=tokens_per_step,
            max_lookahead=max_lookahead,
            kv_window=kv_window,
            apd_mixture_weight=apd_mixture_weight,
            n_parallel_samples=n_parallel_samples,
            max_unmask=max_unmask,
            num_steps=num_steps,
            max_gen_toks=512 if task_name=="math" else 256,
            verifier_ckpt=verifier_ckpt,
        )
        
    elif model_alias == "llada":
        if args.llada_ckpt is None:
            raise ValueError("--llada_ckpt is required when --model_alias=llada")
        llada = AutoModel.from_pretrained(args.llada_ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
        tokenizer = AutoTokenizer.from_pretrained(args.llada_ckpt, trust_remote_code=True)
        model = LladaEvalHarness(pretrained=llada, tokenizer=tokenizer, alg=alg, tokens_per_step=tokens_per_step, num_steps=num_steps)

    else:
        raise ValueError(f"Unknown model alias: {model_alias}. Must be one of 'qwen7b', 'qwen0.5b', 'dream'.")

    return model

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args):
    # Validate algorithm choices based on model alias
    valid_algs = {
        "llada": ["low_confidence", "leftright", "random"],
        "dream": ["leftright", "apd", "entropy", "origin"],
        "qwen7b": ["leftright", None],
        "qwen0.5b": ["leftright", None]
    }
    
    if args.model_alias in valid_algs:
        if args.alg not in valid_algs[args.model_alias]:
            valid_alg_str = ", ".join([str(alg) for alg in valid_algs[args.model_alias]])
            raise ValueError(f"Invalid algorithm '{args.alg}' for model '{args.model_alias}'. "
                           f"Valid algorithms for {args.model_alias}: {valid_alg_str}")
    
    model = get_model(args)
    output_dir = os.path.join(f"{args.output_dir}_{args.task}_{args.verifier_size}")
    os.makedirs(output_dir, exist_ok=True)
    
    # max_length is defined only for generating with AR models
    # it specifies the max length of (input + generated) tokens
    max_length = 16384
    if args.model_alias == "qwen7b" and 'Math' in args.qwen_7b_ckpt:
        max_length = 4096
    elif args.model_alias == "qwen0.5b" and 'Math' in args.qwen_small_ckpt:
        max_length = 4096
    
    task_str = ""
    if args.alg is not None:
        task_str += f"_{args.alg}"
    # Label outputs with canonical flags only
    if args.tokens_per_step is not None:
        task_str += f"_K={args.tokens_per_step}"
    if args.max_lookahead is not None:
        task_str += f"_M={args.max_lookahead}"
    if args.kv_window is not None:
        task_str += f"_W={args.kv_window}"
    if args.apd_mixture_weight is not None:
        task_str += f"_R={args.apd_mixture_weight}"
    if args.n_parallel_samples is not None:
        task_str += f"_N={args.n_parallel_samples}"
    if args.max_unmask is not None:
        if args.max_lookahead is not None:
            assert args.max_unmask <= args.max_lookahead, "max_unmask should be less than or equal to max_lookahead"
        task_str += f"_U={args.max_unmask}"
    if args.num_steps is not None:
        task_str += f"_num_steps={args.num_steps}"
    if args.verifier_size is not None and args.alg=='apd':
        task_str += f"_ar{args.verifier_size}"
    if args.temperature is not None:
        task_str += f"_temp{args.temperature}"
    if args.top_p is not None:
        task_str += f"_topp{args.top_p}"
    if 'qwen' in args.model_alias and args.max_length is not None:
        task_str += f"_genlen{args.max_length}"
        max_length = args.max_length
    if args.tag:
        task_str += f"_{args.tag}"
    output_filename = f"{args.model_alias}_{task_str}_limit{args.limit}.json"
    output_path = os.path.join(output_dir, output_filename)
    logger.info(f"Results will be saved to: {output_path}")
    
    
    task_name = args.task
    
    if task_name == "math":
        system_instruction = "You are a helpful assistant. Justify your final answer by first explaining your step-by-step derivation or reasoning. Conclude by presenting the final answer in the format: boxed{ANSWER}."
    elif task_name == "gpqa":
        system_instruction = "You are a helpful assistant. Justify your final answer by first explaining your step-by-step derivation or reasoning. Conclude by presenting the final answer in the format: (LETTER)."
    else:
        system_instruction = "You are a helpful assistant."
        
    if "qwen" in args.model_alias:
        system_instruction = "You are a helpful assistant." # Normal prompt for qwen models

    task = [TASKS[task_name]]
    
    gen_kwargs = None
    if "qwen" in args.model_alias:
        gen_kwargs = ""
        do_sample = False
        if args.temperature is not None:
            gen_kwargs += f"temperature={args.temperature},"
            do_sample = True
        if args.top_p is not None:
            gen_kwargs += f"top_p={args.top_p},"
            do_sample = True
        gen_kwargs += f"do_sample={do_sample},"
        gen_kwargs += f"max_length={max_length},"
        print(f"max_length is set to {max_length} for generation.")
    
    results = evaluator.simple_evaluate(
        model=model,
        tasks=task,
        batch_size=1,
        limit=args.limit,
        log_samples=True,    
        write_out=True,    
        num_fewshot=0, 
        apply_chat_template=True,
        system_instruction=system_instruction,
        gen_kwargs=gen_kwargs,
        confirm_run_unsafe_code=True
    )

    results["profile"] = model.get_profile()
    if "num_accepted" in results["profile"]:
        num_accepted = results["profile"]["num_accepted"]
        results["profile"].pop("num_accepted")
        save_path = output_path.replace(".json", "_accept_hist.png")
        plot_accept_counts(num_accepted, save_path)
    
    parsed_results = parse_results(results, task_name=task_name)
    
    with open(output_path, 'w') as f:
        json.dump(parsed_results, f, indent=4)
    
def plot_accept_counts(num_accepted, save_path):
    from collections import Counter
    import matplotlib.pyplot as plt

    counts = Counter(num_accepted)
    total = len(num_accepted)

    xs = sorted(counts)
    ys = [counts[x] / total * 100 for x in xs]

    plt.figure()
    plt.bar(xs, ys)
    plt.xlabel("Num Accepted")
    plt.ylabel("Percentage (%)")
    plt.title("Histogram of Num Accepted (%)")
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    main()
    
    