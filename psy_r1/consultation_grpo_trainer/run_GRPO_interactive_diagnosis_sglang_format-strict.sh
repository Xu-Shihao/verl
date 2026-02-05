#!/usr/bin/env bash
# GRPO 交互式问诊训练脚本 - 启用 Multi-turn
set -xeuo pipefail

# # 设置环境proxy
# export http_proxy="http://cloudml:gP1dY0uI0o@10.119.176.202:3128"
# export https_proxy="http://cloudml:gP1dY0uI0o@10.119.176.202:3128"

# 设置WANDB
export WANDB_MODE=online
export WANDB_API_KEY=d8e131b9817bc59353326755d6db8b705a4d8d4d
wandb login --relogin $WANDB_API_KEY
export WANDB_ENTITY="shihao-xu-ntu"

# 设置CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 设置日志路径
set -x

# 创建日志路径
LOG_DIR="/tcci_mnt/shihao/project/verl/psy_r1/logs"
mkdir -p $LOG_DIR

# 读取现在时间
NOW=$(date +%Y%m%d_%H%M%S)

HOME="/tcci_mnt/shihao/project/verl"

# 模型名称
MODEL_PATH="/tcci_mnt/shihao/models/Qwen3-8B"
MODEL_BASE_NAME="qwen3-8B_interactive_diagnosis_multiturn_ds_fs"
NNODES=1

# Patient Agent 配置
PATIENT_AGENT_URL="http://192.168.5.187:8001"
PATIENT_MODEL="Qwen3-1.7B"

# 项目配置
project_name='SMHC_Interactive_Diagnosis_MultiTurn_RL'
exp_name=grpo_${MODEL_BASE_NAME}

# 确保不连接远程Ray集群
unset RAY_ADDRESS
export RAY_DISABLE_IMPORT_WARNING=1

# 算法配置 - GRPO标准配置
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001
kl_loss_type=low_var_kl

# 数据配置
max_prompt_length=2048
max_response_length=2048  # 增加以支持多轮对话

# 训练配置
train_prompt_bsz=64  # 减小批次以适应多轮交互
train_prompt_mini_bsz=16
n_resp_per_prompt=5  # 减少每个prompt的响应数
n_gpus_per_node=8
tensor_model_parallel_size=1

# 性能相关参数
infer_micro_batch_size=2  # 多轮交互时减小
train_micro_batch_size=2
offload=False

# 采样参数
temperature=0.8
top_p=0.95
top_k=-1
entropy_coeff=0

# 路径配置
CKPTS_DIR="/tcci_mnt/shihao/project/verl/checkpoints/${project_name}/${exp_name}"
TRAIN_FILE="$HOME/psy_r1/LingxiDiag-interactive-diagnosis/train.parquet"
VAL_FILE="$HOME/psy_r1/LingxiDiag-interactive-diagnosis/val.parquet"

# 交互式问诊特定参数
USE_INTERACTIVE_DIAGNOSIS=True
MAX_DIALOGUE_TURNS=50
EFFICIENCY_WEIGHT=0.1

# ICD奖励参数
USE_ICD_REWARD=True

# Tool 配置文件路径（使用启用严格格式检查的配置）
TOOL_CONFIG="${HOME}/psy_r1/consultation_grpo_trainer/patient_tool_config_strict.yaml"

# 创建检查点目录
mkdir -p "${CKPTS_DIR}"

# 切换到工作目录
cd "${HOME}"

# 启动 GRPO 交互式问诊训练（Multi-turn 模式）
echo "========================================="
echo "GRPO 交互式问诊训练 (Multi-turn 模式)"
echo "========================================="
echo "模型: ${MODEL_PATH}"
echo "Patient Agent: ${PATIENT_AGENT_URL}"
echo "训练数据: ${TRAIN_FILE}"
echo "验证数据: ${VAL_FILE}"
echo "最大对话轮数: ${MAX_DIALOGUE_TURNS}"
echo "Multi-turn: ENABLED"
echo "========================================="

HYDRA_FULL_ERROR=1 && python3 -m psy_r1.verl.trainer.main_ppo_psy \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.return_raw_chat=True \
    +data.need_tools_kwargs=True \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_micro_batch_size} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_DIALOGUE_TURNS} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_DIALOGUE_TURNS} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
    actor_rollout_ref.rollout.multi_turn.format=llama3_json \
    actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=False \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
    +actor_rollout_ref.rollout.multi_turn.dynamic_sampling.enable=True \
    +actor_rollout_ref.rollout.multi_turn.dynamic_sampling.oversample_ratio=1.2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    +reward_model.use_icd_reward=${USE_ICD_REWARD} \
    +reward_model.patient_agent_url="${PATIENT_AGENT_URL}" \
    +reward_model.patient_model_name="${PATIENT_MODEL}" \
    +reward_model.max_dialogue_turns=${MAX_DIALOGUE_TURNS} \
    +reward_model.efficiency_weight=${EFFICIENCY_WEIGHT} \
    reward_model.show_training_examples=True \
    reward_model.show_val_examples=True \
    trainer.critic_warmup=0.0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.total_epochs=20 \
    trainer.val_before_train=False \
    trainer.default_local_dir="${CKPTS_DIR}" \
    $@ 2>&1 | tee $LOG_DIR/${exp_name}_$NOW.log

