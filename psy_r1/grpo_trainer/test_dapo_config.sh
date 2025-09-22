#!/usr/bin/env bash
# 测试DAPO配置的简化脚本

# 设置环境变量
export https_proxy=http://10.119.16.227:7890 http_proxy=http://10.119.16.227:7890 all_proxy=socks5://10.119.16.227:7890
export WANDB_API_KEY=d8e131b9817bc59353326755d6db8b705a4d8d4d
export CUDA_VISIBLE_DEVICES=4,5,6,7

HOME="/mnt/tcci/shihao/project/verl"
MODEL_PATH="/mnt/tcci/shihao/models/Qwen3-1.7B"
N_GPUS=4

# 只运行配置验证，不实际训练
echo "Testing DAPO configuration..."

HYDRA_FULL_ERROR=1 python3 -m recipe.dapo.main_dapo \
    --config-path=/dev/null \
    --config-name=dapo_trainer \
    data.train_files=$HOME/psy_r1/SMHC_data_v4/train.parquet \
    data.val_files=$HOME/psy_r1/SMHC_data_v4/val.parquet \
    data.reward_fn_key=data_source \
    data.train_batch_size=64 \
    data.gen_batch_size=192 \
    data.max_prompt_length=6144 \
    data.max_response_length=3096 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    reward_model.reward_manager=dapo \
    +reward_model.custom_reward_function.path=$HOME/psy_r1/verl/utils/dapo_reward_score_psy.py \
    +reward_model.custom_reward_function.name=register_psy_reward_function \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    --help
