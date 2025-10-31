#!/bin/bash
export PYTHONPATH='.'

base_model='openlm-research/open_llama_7b_v2' #'baffo32/decapoda-research-llama-7B-hf'
#base_model=$1 # e.g., decapoda-research/llama-7b-hf， baffo32/decapoda-research-llama-7B-hf， 
CUDA_VISIBLE_DEVICES=4 python lm-evaluation-harness/main.py --model hf --model_args pretrained=$base_model    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq  --no_cache

echo "[START] - Start Test MAC"
CUDA_VISIBLE_DEVICES=4 python test_speedup.py --base_model   $base_model --model_type pretrain  --ckpt  prune_log/$prune_ckpt_path/pytorch_model.bin

