#!/usr/bin/env bash
# ============================================================================
# GRPO 交互式问诊训练脚本 - 统一版本
# ============================================================================
# 支持以下可配置功能：
#   1. 严格格式检查 (strict_format)
#   2. 梯形长度奖励函数 (length_reward)
#   3. SIG过程奖励 (sig_reward) - Shapley Information Gain
#
# 使用方法：
#   # 默认配置（关闭所有额外功能）
#   ./run_GRPO_interactive_diagnosis_unified.sh
#
#   # 启用严格格式检查
#   ./run_GRPO_interactive_diagnosis_unified.sh --strict-format
#
#   # 启用长度奖励
#   ./run_GRPO_interactive_diagnosis_unified.sh --length-reward
#
#   # 同时启用两项功能
#   ./run_GRPO_interactive_diagnosis_unified.sh --strict-format --length-reward
#
#   # 自定义长度奖励参数
#   ./run_GRPO_interactive_diagnosis_unified.sh --length-reward --length-weight 0.3
#
#   # 启用SIG过程奖励（Shapley Information Gain）
#   ./run_GRPO_interactive_diagnosis_unified.sh --sig-reward
#
#   # 自定义SIG参数
#   ./run_GRPO_interactive_diagnosis_unified.sh --sig-reward --sig-weight 0.5 --sig-monte-carlo-k 50
#
#   # 同时启用所有功能
#   ./run_GRPO_interactive_diagnosis_unified.sh --strict-format --length-reward --sig-reward
#
# ============================================================================

set -xeuo pipefail

# ============================================================================
# 解析命令行参数
# ============================================================================
USE_STRICT_FORMAT_CHECK=false
USE_LENGTH_REWARD=false
LENGTH_REWARD_WEIGHT=0.2
LENGTH_MIN_TURNS=10
LENGTH_OPTIMAL_START=15
LENGTH_OPTIMAL_END=25
LENGTH_MAX_TURNS=50
CUSTOM_SUFFIX="_v3"

# SIG (Shapley Information Gain) 奖励参数
USE_SIG_REWARD=false
SIG_REWARD_WEIGHT=0.5
SIG_CORRECTNESS_BONUS_WEIGHT=0.3
SIG_MONTE_CARLO_K=50
SIG_LLM_BASE_URL="http://localhost:8000/v1"

# ============================================================================
# 环境配置
# ============================================================================

# 设置WANDB
export WANDB_MODE=online
export WANDB_API_KEY=d8e131b9817bc59353326755d6db8b705a4d8d4d
wandb login --relogin $WANDB_API_KEY
export WANDB_ENTITY="shihao-xu-ntu"

# 设置CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 创建日志路径
LOG_DIR="/tcci_mnt/shihao/project/verl/psy_r1/logs"
mkdir -p $LOG_DIR

# 读取现在时间
NOW=$(date +%Y%m%d_%H%M%S)

HOME="/tcci_mnt/shihao/project/verl"

# ============================================================================
# 模型和数据配置
# ============================================================================

MODEL_PATH="/tcci_mnt/shihao/outputs/toocall_interactive/qwen3-8B_interactive_toolcall_lora-sft_lingxidiag-16k_v3_0210"
MODEL_BASE_NAME="qwen3-8B_interactive_diagnosis${CUSTOM_SUFFIX}"
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

# ============================================================================
# 算法配置 - GRPO标准配置
# ============================================================================

adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001
kl_loss_type=low_var_kl

# ============================================================================
# 数据配置
# ============================================================================

max_prompt_length=2048
max_response_length=10240  # 支持多轮对话

# ============================================================================
# 训练配置
# ============================================================================

train_prompt_bsz=64
train_prompt_mini_bsz=16
n_resp_per_prompt=5
n_gpus_per_node=8
tensor_model_parallel_size=1

# 性能相关参数
infer_micro_batch_size=2
train_micro_batch_size=2
offload=False

# 采样参数
temperature=0.8
top_p=0.95
top_k=-1
entropy_coeff=0

# ============================================================================
# 路径配置
# ============================================================================

CKPTS_DIR="/tcci_mnt/shihao/project/verl/checkpoints/${project_name}/${exp_name}"
TRAIN_FILE="$HOME/psy_r1/dataset_rl/LingxiDiag-interactive-long-empathy-v2/train.parquet"
VAL_FILE="$HOME/psy_r1/dataset_rl/LingxiDiag-interactive-long-empathy-v2/val.parquet"

# ============================================================================
# 交互式问诊参数
# ============================================================================

USE_INTERACTIVE_DIAGNOSIS=True
MAX_DIALOGUE_TURNS=50
USE_ICD_REWARD=True

# ============================================================================
# 动态生成 Tool 配置文件
# ============================================================================

TOOL_CONFIG_DIR="${HOME}/psy_r1/consultation_grpo_trainer"
TOOL_CONFIG="${TOOL_CONFIG_DIR}/patient_tool_config_runtime_${NOW}.yaml"

cat > "${TOOL_CONFIG}" << EOF
# 运行时动态生成的 Tool 配置文件
# 生成时间: ${NOW}
# 配置选项:
#   - use_strict_format_check: ${USE_STRICT_FORMAT_CHECK}
#   - use_length_reward: ${USE_LENGTH_REWARD}
#   - length_reward_weight: ${LENGTH_REWARD_WEIGHT}
#   - length_reward_config: [${LENGTH_MIN_TURNS}, ${LENGTH_OPTIMAL_START}, ${LENGTH_OPTIMAL_END}, ${LENGTH_MAX_TURNS}]
#   - use_sig_reward: ${USE_SIG_REWARD}
#   - sig_reward_weight: ${SIG_REWARD_WEIGHT}
#   - sig_correctness_bonus_weight: ${SIG_CORRECTNESS_BONUS_WEIGHT}
#   - sig_monte_carlo_k: ${SIG_MONTE_CARLO_K}

tools:
  - class_name: "psy_r1.verl.tools.patient_interaction_tools.AskPatientTool"
    config:
      type: native
      patient_agent_url: "${PATIENT_AGENT_URL}"
      default_patient_version: "v3"
      default_model_name: "Qwen3-32B"
      timeout: 120
      max_history: 15
      log_payload: false
      log_messages: false
      log_complete_dialogue: true
      max_retries: 3
      retry_delay: 2.0
      retry_backoff: 1.5
    tool_schema:
      type: "function"
      function:
        name: "ask_patient"
        description: "向本地 Patient Agent 提问并获取回答。"
        parameters:
          type: "object"
          properties:
            question:
              type: "string"
              description: "需要向患者提出的问题。"
            patient_id:
              type: "string"
              description: "可选，覆盖默认患者 ID。"
            patient_version:
              type: "string"
              description: "可选，覆盖患者版本（默认 v3）。"
            model_name:
              type: "string"
              description: "可选，Patient Agent 使用的底模名称。"
          required: ["question"]

  - class_name: "psy_r1.verl.tools.patient_interaction_tools.DiagnoseRewardTool"
    config:
      type: native
      default_patient_version: "v3"
      log_complete_dialogue: true
      # 格式检查配置
      use_strict_format_check: ${USE_STRICT_FORMAT_CHECK}
      # 长度奖励配置
      use_length_reward: ${USE_LENGTH_REWARD}
      length_reward_weight: ${LENGTH_REWARD_WEIGHT}
      length_reward_config:
        min_turns: ${LENGTH_MIN_TURNS}
        optimal_start: ${LENGTH_OPTIMAL_START}
        optimal_end: ${LENGTH_OPTIMAL_END}
        max_turns: ${LENGTH_MAX_TURNS}
      # SIG (Shapley Information Gain) 过程奖励配置
      use_sig_reward: ${USE_SIG_REWARD}
      sig_reward_config:
        sig_reward_weight: ${SIG_REWARD_WEIGHT}
        correctness_bonus_weight: ${SIG_CORRECTNESS_BONUS_WEIGHT}
        monte_carlo_k: ${SIG_MONTE_CARLO_K}
        llm_base_url: "${SIG_LLM_BASE_URL}"
        enable_caching: true
        cache_shapley_values: true
        log_sig_details: true
    tool_schema:
      type: "function"
      function:
        name: "do_diagnose"
        description: "提交最终 ICD-10 诊断并计算奖励。必须在 diagnosis 参数中提供完整的诊断推理过程和 ICD 代码。"
        parameters:
          type: "object"
          properties:
            diagnosis:
              type: "string"
              description: "完整的诊断文本，必须包含：1) <think>诊断推理过程</think> 2) <box>ICD-10代码</box>。例如：<think>患者表现为持续情绪低落...</think> <box>F32.1</box>"
            patient_id:
              type: "string"
              description: "可选，覆盖患者 ID。"
            patient_version:
              type: "string"
              description: "可选，覆盖患者版本。"
          required: ["diagnosis"]
EOF

# 创建检查点目录
mkdir -p "${CKPTS_DIR}"

# 切换到工作目录
cd "${HOME}"

# ============================================================================
# 打印配置信息
# ============================================================================

echo "============================================================================"
echo "GRPO 交互式问诊训练 - 统一版本"
echo "============================================================================"
echo "模型: ${MODEL_PATH}"
echo "Patient Agent: ${PATIENT_AGENT_URL}"
echo "训练数据: ${TRAIN_FILE}"
echo "验证数据: ${VAL_FILE}"
echo "最大对话轮数: ${MAX_DIALOGUE_TURNS}"
echo ""
echo "功能开关:"
if [ "${USE_STRICT_FORMAT_CHECK}" = "true" ]; then
    echo "  - 严格格式检查: ENABLED"
    echo "    (检查 think/tool_call/diagnose 三项格式要求)"
else
    echo "  - 严格格式检查: DISABLED (仅检查 <box> 标签)"
fi
if [ "${USE_LENGTH_REWARD}" = "true" ]; then
    echo "  - 长度奖励: ENABLED (weight=${LENGTH_REWARD_WEIGHT})"
    echo "    最优轮数区间: ${LENGTH_OPTIMAL_START}-${LENGTH_OPTIMAL_END}"
    echo "    有效轮数范围: ${LENGTH_MIN_TURNS}-${LENGTH_MAX_TURNS}"
else
    echo "  - 长度奖励: DISABLED"
fi
if [ "${USE_SIG_REWARD}" = "true" ]; then
    echo "  - SIG过程奖励: ENABLED (weight=${SIG_REWARD_WEIGHT})"
    echo "    正确性奖励权重: ${SIG_CORRECTNESS_BONUS_WEIGHT}"
    echo "    Monte Carlo采样次数: ${SIG_MONTE_CARLO_K}"
    echo "    LLM API: ${SIG_LLM_BASE_URL}"
else
    echo "  - SIG过程奖励: DISABLED"
fi
echo ""
echo "Tool 配置文件: ${TOOL_CONFIG}"
echo "实验名称: ${exp_name}"
echo "============================================================================"

# ============================================================================
# 启动训练
# ============================================================================

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
    actor_rollout_ref.rollout.max_num_batched_tokens=16000 \
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

# 清理临时配置文件（可选，注释掉以保留调试）
# rm -f "${TOOL_CONFIG}"

echo "============================================================================"
echo "训练完成"
echo "日志文件: $LOG_DIR/${exp_name}_$NOW.log"
echo "============================================================================"
