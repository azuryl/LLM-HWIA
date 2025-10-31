prune_ckpt_path='llama_prune_0.25_block_wise_param_fusion_baichuan_N50'
tune_ckpt_path='llama_0.25_param_fusion_block_wise_baichuan_N50'
base_model='baichuan-inc/Baichuan-7B'
prune_ratio='0.25'
num_examples='50'
nGPU='7'
echo "GPU is $nGPU"


echo "[START] - Start Pruning Model"
#CUDA_VISIBLE_DEVICES=0 python hf_prune.py --pruning_ratio 0.20 --device cpu  --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --test_after_train --taylor param_second --save_model 
#CUDA_VISIBLE_DEVICES=$nGPU python baichuan.py --pruning_ratio 0.25 --device cpu  --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor  --test_after_train --taylor param_second --save_model --num_examples 50 --evratio 0.6 --base_model $base_model
echo "[FINISH] - Finish Pruning Model"

echo "[START] - Start Tuning"
export PYTHONPATH='/home/azuryl/.cache/huggingface/modules:$PYTHONPATH'
CUDA_VISIBLE_DEVICES=$nGPU python post_training.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir tune_log/$tune_ckpt_path --wandb_project llama_tune --lora_r 8 --lora_target_modules gate_proj,down_proj,up_proj,W_pack,o_proj,W_pack,o_proj --num_epochs 2 --learning_rate 1e-4 --batch_size 64
echo "[FINISH] - Finish Prune and Post-Training."
echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$tune_ckpt_path}/"

echo "You can use the command:"
echo "       python generate.py --model_type tune_prune_LLM --ckpt prune_log/$prune_ckpt_path/pytorch_model.bin --lora_ckpt tune_log/$tune_ckpt_path"
echo "to use the pruned model"

echo "[START] - Start Evaluating"
CUDA_VISIBLE_DEVICES=$nGPU bash scripts/evaluate.sh $base_model tune_log/$tune_ckpt_path prune_log/$prune_ckpt_path/  200

echo "[START] - Start Test MAC"
python test_speedup.py --base_model   $base_model --model_type pruneLLM  --ckpt  prune_log/$prune_ckpt_path/pytorch_model.bin

