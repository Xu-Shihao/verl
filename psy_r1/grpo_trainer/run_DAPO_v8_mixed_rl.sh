#!/usr/bin/env bash
# ============================================================
# v8 DAPO脚本 - 混合RL训练 (binary + multiclass + recommendation)
# 使用 SMHC_data_v8 数据集和 v8 混合奖励函数
# ============================================================
set -xeuo pipefail

# 设置WANDB
export WANDB_API_KEY=d8e131b9817bc59353326755d6db8b705a4d8d4d
wandb login --relogin $WANDB_API_KEY
export WANDB_ENTITY="shihao-xu-ntu"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

set -x

# 日志路径
LOG_DIR="/tcci_mnt/shihao/project/verl/psy_r1/logs"
mkdir -p $LOG_DIR

NOW=$(date +%Y%m%d_%H%M%S)

HOME="/tcci_mnt/shihao/project/verl"

# ============================================================
# 模型配置 (请根据实际情况修改)
# ============================================================
MODEL_PATH="/tcci_mnt/shihao/outputs/dataset_v2/qwen3-8B_auxiliary_diagnosis_lora-sft_reasoning_kimi-k2-0905_v7_lr1e-6"
MODEL_BASE_NAME="qwen3-8B-sft"
NNODES=1

# ============================================================
# 项目配置
# ============================================================
project_name='SMHC_ICD_recommendation_RL'
exp_name=dapo_${MODEL_BASE_NAME}_v8_mixed_lr5e-7

# # 确保不连接远程Ray集群
# unset RAY_ADDRESS
# export RAY_DISABLE_IMPORT_WARNING=1

# ============================================================
# 算法配置 - DAPO
# ============================================================
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001
kl_loss_type=low_var_kl

# DAPO特有参数
clip_ratio_low=0.2
clip_ratio_high=0.28
clip_ratio_c=10.0

# ============================================================
# 数据配置
# ============================================================
max_prompt_length=6144
max_response_length=6144
enable_overlong_buffer=True
overlong_buffer_len=512
overlong_penalty_factor=1.0

# ============================================================
# 训练配置
# ============================================================
loss_agg_mode="token-mean"
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=15
train_prompt_bsz=128
gen_prompt_bsz=$((train_prompt_bsz * 3))
train_prompt_mini_bsz=32
n_resp_per_prompt=5
n_gpus_per_node=8

# 性能参数
use_dynamic_bsz=True
infer_micro_batch_size=4
train_micro_batch_size=4
offload=False
sp_size=1
gen_tp=1

# 采样参数
temperature=1.0
top_p=1.0
top_k=-1
entropy_coeff=0

# ============================================================
# 路径配置 - 使用 v8 数据
# ============================================================
CKPTS_DIR="/tcci_mnt/shihao/project/verl/checkpoints/${project_name}/${exp_name}"
TRAIN_FILE="$HOME/psy_r1/SMHC_data_v8/train.parquet"
VAL_FILE="$HOME/psy_r1/SMHC_data_v8/val.parquet"

mkdir -p "${CKPTS_DIR}"

cd "${HOME}"

# ============================================================
# 启动 v8 DAPO 训练
# 使用 main_dapo_v8 而非 main_dapo_psy
# 奖励函数已内置在 v8 trainer 中，根据 data_source 自动分发
# 同时使用 DAPO reward manager + v8 compute_score 作为 custom_reward_function
# ============================================================
HYDRA_FULL_ERROR=1 && python3 -m psy_r1.verl.trainer.main_dapo_v8 \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.reward_fn_key=data_source \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_micro_batch_size} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.max_num_batched_tokens=13240 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${infer_micro_batch_size} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    reward_model.reward_manager=dapo \
    +reward_model.custom_reward_function.path=$HOME/psy_r1/verl/utils/v8_mixed_reward.py \
    +reward_model.custom_reward_function.name=compute_score_v8 \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    reward_model.show_training_examples=True \
    reward_model.show_val_examples=True \
    trainer.critic_warmup=0.1 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=20 \
    trainer.val_before_train=True \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    ray_init.num_cpus=32 \
    $@ 2>&1 | tee $LOG_DIR/${exp_name}_$NOW.log
