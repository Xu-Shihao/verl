#!/usr/bin/env bash
# DAPO脚本 - 心理诊断任务 (直接运行版本)
# 基于GRPO脚本适配到DAPO算法，支持1个node的8卡运行

# 设置环境proxy
export https_proxy=http://10.119.16.227:7890 http_proxy=http://10.119.16.227:7890 all_proxy=socks5://10.119.16.227:7890

wandb login --relogin $WANDB_API_KEY

# 设置日志路径
set -x

# 创建日志路径
LOG_DIR="/mnt/tcci/shihao/project/verl/psy_r1/logs"
mkdir -p $LOG_DIR

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# MODEL_PATH="/mnt/tcci/shihao/outputs/qwen3-8B_auxiliary_diagnosis_full_sft_ds-r1_v2"
# MODEL_BASE_NAME="qwen3-8B"

MODEL_PATH="/mnt/tcci/shihao/models/Qwen3-1.7B"
MODEL_BASE_NAME="qwen3-1.7B"
N_GPUS=8  # 使用8个GPU


# 读取现在时间
NOW=$(date +%Y%m%d_%H%M%S)

HOME="/mnt/tcci/shihao/project/verl"

HYDRA_FULL_ERROR=1 && python3 -m psy_r1.verl.trainer.main_dapo_psy \
    data.train_files=$HOME/psy_r1/SMHC_data_v4/train.parquet \
    data.val_files=$HOME/psy_r1/SMHC_data_v4/val.parquet \
    data.reward_fn_key=data_source \
    data.train_batch_size=128 \
    data.gen_batch_size=384 \
    data.max_prompt_length=6144 \
    data.max_response_length=3096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=acc \
    algorithm.filter_groups.max_num_gen_batches=10 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=True \
    reward_model.overlong_buffer.len=512 \
    reward_model.overlong_buffer.penalty_factor=1.0 \
    trainer.critic_warmup=0.1 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='SMHC_DAPO_diagnosis_with_reasoning' \
    trainer.experiment_name=dapo_${MODEL_BASE_NAME}_reasoning_sft_ds-r1_v2_8_gpu \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
    trainer.val_before_train=False \
    ray_init.num_cpus=32 \
    $@ 2>&1 | tee $LOG_DIR/dapo_${MODEL_BASE_NAME}_diagnosis_reasoning_direct_$NOW.log
