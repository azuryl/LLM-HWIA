lora_r='256'
prune_ckpt_path='open_llama_3b_v2_prune_0.2_block_wise_param_second_N50'
tune_ckpt_path='open_llama_3b_v2_0.2_param_second_block_wise_N50_Dora_r256'
num_examples='50'
base_model='openlm-research/open_llama_3b_v2'
prune_ratio='0.25'
nGPU='6'
epoch='1400'
evratio='1'

echo "GPU is $nGPU"
echo "lora_r= $lora_r"
echo "[START] - Start Pruning Model"
#CUDA_VISIBLE_DEVICES=0 python hf_prune.py --pruning_ratio 0.20 --device cpu  --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --test_after_train --taylor param_second --save_model 
#CUDA_VISIBLE_DEVICES=$nGPU python hf_prune.py --pruning_ratio $prune_ratio --device cpu  --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 23 --block_attention_layer_start 4 --block_attention_layer_end 23 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor  --test_after_train --taylor param_second --save_model --num_examples $num_examples --evratio $evratio --base_model $base_model
echo "[FINISH] - Finish Pruning Model"

echo "[START] - Start Tuning"

#CUDA_VISIBLE_DEVICES=$nGPU python post_training.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir tune_log/$tune_ckpt_path --wandb_project llama_tune --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64
CUDA_VISIBLE_DEVICES=$nGPU python fine-tuning.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir tune_log/$tune_ckpt_path --wandb_project llama_tune --lora_r $lora_r --num_epochs 2 --learning_rate 1e-4 --batch_size 64   --use_dora True  --use_fp16 True --use_rslora False
#CUDA_VISIBLE_DEVICES=$nGPU python dora_fine-tuning.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir tune_log/$tune_ckpt_path --wandb_project llama_tune --lora_r $lora_r --num_epochs 2 --learning_rate 4e-4 --batch_size 64   --use_dora False  --use_rslora False
echo "[FINISH] - Finish Prune and Post-Training."
echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$tune_ckpt_path}/"

echo "You can use the command:"
echo "       python generate.py --model_type tune_prune_LLM --ckpt prune_log/$prune_ckpt_path/pytorch_model.bin --lora_ckpt tune_log/$tune_ckpt_path"
echo "to use the pruned model"

echo "[START] - Start Evaluating"
CUDA_VISIBLE_DEVICES=$nGPU bash scripts/evaluate.sh $base_model tune_log/$tune_ckpt_path prune_log/$prune_ckpt_path  $epoch
echo "[FINISH] - Finish  Evaluating"

echo "[START] - Start Test MAC"
python test_speedup.py --base_model   $base_model --model_type pruneLLM  --ckpt  prune_log/$prune_ckpt_path/pytorch_model.bin

