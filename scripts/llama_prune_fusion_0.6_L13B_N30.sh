prune_ckpt_path='llama_prune_0.2_block_wise_param_second_fusion0.6_L13B_N30'
tune_ckpt_path='llama_0.2_param_second_block_wise_fusion0.6_L13B_N30'

echo "[START] - Start Pruning Model"
#CUDA_VISIBLE_DEVICES=0 python hf_prune.py --pruning_ratio 0.20 --device cpu  --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --test_after_train --taylor param_second --save_model 
CUDA_VISIBLE_DEVICES=7 python hf_prune.py --pruning_ratio 0.20 --device cpu  --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor  --test_after_train --taylor param_second --save_model --num_examples 30 --evratio 0.6 --base_model NousResearch/Llama-2-13b-hf
echo "[FINISH] - Finish Pruning Model"

echo "[START] - Start Tuning"
CUDA_VISIBLE_DEVICES=7 python post_training.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir tune_log/$tune_ckpt_path --wandb_project llama_tune --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64
echo "[FINISH] - Finish Prune and Post-Training."
echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$tune_ckpt_path}/"

echo "You can use the command:"
echo "       python generate.py --model_type tune_prune_LLM --ckpt prune_log/$prune_ckpt_path/pytorch_model.bin --lora_ckpt tune_log/$tune_ckpt_path"
echo "to use the pruned model"

echo "[START] - Start Evaluating"
CUDA_VISIBLE_DEVICES=7 bash scripts/evaluate.sh NousResearch/Llama-2-13b-hf tune_log/llama_0.2_param_second_block_wise_fusion0.6_L13B_N30 prune_log/llama_prune_0.2_block_wise_param_second_fusion0.6_L13B_N30/  200

