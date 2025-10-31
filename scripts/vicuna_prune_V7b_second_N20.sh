prune_ckpt_path='vicuna_prune_0.2_block_wise_param_second_V7B_N20'
tune_ckpt_path='vicuna_0.2_param_second_block_wise_V7B_N20'
base_model='lmsys/vicuna-7b-v1.5'
nGPU='2'
num_examples='20'

echo "GPU is $nGPU"

echo "[START] - Start Pruning Model"
#CUDA_VISIBLE_DEVICES=0 python hf_prune.py --pruning_ratio 0.20 --device cpu  --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --test_after_train --taylor param_second --save_model 
CUDA_VISIBLE_DEVICES=$nGPU python hf_prune.py --pruning_ratio 0.20 --device cpu  --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor  --test_after_train --taylor param_second --save_model --num_examples $num_examples --evratio 1 --base_model lmsys/vicuna-7b-v1.5
echo "[FINISH] - Finish Pruning Model"

echo "[START] - Start Tuning"
CUDA_VISIBLE_DEVICES=$nGPU python post_training.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir tune_log/$tune_ckpt_path --wandb_project llama_tune --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64
echo "[FINISH] - Finish Prune and Post-Training."
echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$tune_ckpt_path}/"

echo "You can use the command:"
echo "       python generate.py --model_type tune_prune_LLM --ckpt prune_log/$prune_ckpt_path/pytorch_model.bin --lora_ckpt tune_log/$tune_ckpt_path"
echo "to use the pruned model"

echo "[START] - Start Evaluating"
CUDA_VISIBLE_DEVICES=$nGPU bash scripts/evaluate.sh $base_model tune_log/$tune_ckpt_path prune_log/$prune_ckpt_path/  200
echo "[FINISH] - Finish  Evaluating"

echo "[START] - Start Test MAC"
python test_speedup.py --base_model   $base_model --model_type pretrain  --ckpt  prune_log/$prune_ckpt_path/pytorch_model.bin
