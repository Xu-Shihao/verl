#!/usr/bin/env bash
# GRPO脚本 - ICD代码推荐任务
set -xeuo pipefail

# # 设置环境proxy
# export http_proxy="http://cloudml:gP1dY0uI0o@10.119.176.202:3128"
# export https_proxy="http://cloudml:gP1dY0uI0o@10.119.176.202:3128"

# 设置WANDB
wandb online
export WANDB_API_KEY=d8e131b9817bc59353326755d6db8b705a4d8d4d
wandb login --relogin $WANDB_API_KEY
export WANDB_ENTITY="shihao-xu-ntu"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 多节点通信：指定正确的网络接口（192.168.0.x 对应 eth1）
export GLOO_SOCKET_IFNAME=eth1
export NCCL_SOCKET_IFNAME=eth1

# 设置日志路径
set -x

# 创建日志路径
LOG_DIR="/tcci_mnt/shihao/project/verl/psy_r1/logs"
mkdir -p $LOG_DIR

# 读取现在时间
NOW=$(date +%Y%m%d_%H%M%S)

HOME="/tcci_mnt/shihao/project/verl"

# 模型名称
MODEL_PATH="/tcci_mnt/shihao/outputs/dataset_v2/qwen3-32B_auxiliary_diagnosis_lora-sft_reasoning_kimi-k2-0905_v7_lr1e-6_0113"
MODEL_BASE_NAME="qwen3-32B-sft_auxiliary_diagnosis_v7_lr1e-6_tp2_full"
NNODES=2

# 项目配置
project_name='SMHC_ICD_recommendation_RL'
exp_name=grpo_${MODEL_BASE_NAME}_icd_recommendation

# # 确保不连接远程Ray集群
# unset RAY_ADDRESS
# export RAY_DISABLE_IMPORT_WARNING=1

# 算法配置 - GRPO标准配置
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001
kl_loss_type=low_var_kl

# 数据配置
max_prompt_length=6144
max_response_length=4800
max_num_batched_tokens=11300

# 训练配置
train_prompt_bsz=128
ppo_mini_batch_size=64
n_resp_per_prompt=5
n_gpus_per_node=8

# 性能相关参数
log_prob_micro_batch_size_per_gpu=4
ppo_micro_batch_size_per_gpu=2
offload=False

# 采样参数
temperature=1.0
top_p=1.0
top_k=-1
entropy_coeff=0

# 路径配置
CKPTS_DIR="/tcci_mnt/shihao/project/verl/checkpoints/${project_name}/${exp_name}"
TRAIN_FILE="$HOME/psy_r1/SMHC_data_v7/train.parquet"
VAL_FILE="$HOME/psy_r1/SMHC_data_v7/val.parquet"

# ICD奖励参数 - 启用ICD奖励（关键配置）
USE_ICD_REWARD=True
USE_SYMPTOM_REWARD=False
SYMPTOM_ALPHA=0.1

# 创建检查点目录
mkdir -p "${CKPTS_DIR}"

# 切换到工作目录
cd "${HOME}"

# 启动GRPO训练 (使用psy_r1模块，启用ICD奖励)
# 注意：GRPO使用main_ppo_psy，但通过reward_model配置来启用ICD奖励
HYDRA_FULL_ERROR=1 && python3 -m psy_r1.verl.trainer.main_ppo_psy \
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
    reward_model.use_psy_reward=True \
    +reward_model.use_icd_reward=${USE_ICD_REWARD} \
    +reward_model.use_symptom_reward=${USE_SYMPTOM_REWARD} \
    +reward_model.symptom_alpha=${SYMPTOM_ALPHA} \
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
