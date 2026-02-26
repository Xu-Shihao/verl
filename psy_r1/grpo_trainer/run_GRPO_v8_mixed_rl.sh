#!/usr/bin/env bash
# ============================================================
# v8 GRPO脚本 - 混合RL训练 (binary + multiclass + recommendation)
# 使用 SMHC_data_v8 数据集和 v8 混合奖励函数
# ============================================================
set -xeuo pipefail

# 设置WANDB
wandb online
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
MODEL_BASE_NAME="qwen3-8B-sft_v7"
NNODES=1

# ============================================================
# 项目配置
# ============================================================
project_name='SMHC_v8_mixed_RL'
exp_name=grpo_${MODEL_BASE_NAME}_v8_mixed

# ============================================================
# 算法配置 - GRPO
# ============================================================
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001
kl_loss_type=low_var_kl

# ============================================================
# 数据配置
# ============================================================
max_prompt_length=6144
max_response_length=6144
max_num_batched_tokens=13240

# ============================================================
# 训练配置
# ============================================================
train_prompt_bsz=128
ppo_mini_batch_size=64
n_resp_per_prompt=5
n_gpus_per_node=8

# 性能参数
log_prob_micro_batch_size_per_gpu=4
ppo_micro_batch_size_per_gpu=4

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
# 启动 v8 GRPO 训练
# 使用 main_ppo_v8 而非 main_ppo_psy
# 奖励函数已内置在 v8 trainer 中，根据 data_source 自动分发
# ============================================================
HYDRA_FULL_ERROR=1 && python3 -m psy_r1.verl.trainer.main_ppo_v8 \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
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
    trainer.val_before_train=False \
    trainer.default_local_dir="${CKPTS_DIR}" \
    $@ 2>&1 | tee $LOG_DIR/${exp_name}_$NOW.log
