#!/usr/bin/env bash
# ============================================================================
# GRPO 交互式问诊训练脚本 - v4 (修复SIG + 防止格式崩溃)
# ============================================================================
# v4 版本关键修复：
#   1. SIG奖励修复：使用单独部署的LLM服务计算SIG
#   2. 增强KL约束：kl_loss_coef从0.001增加到0.01，防止格式崩溃
#   3. 添加格式惩罚：启用strict_format_check，对格式错误进行惩罚
#   4. 使用更强的Patient Model (Qwen3-8B)
#   5. SIG需要patient_info字段，暂时在DiagnoseRewardTool中动态获取
#
# 对比 v3_sig 的修改：
#   - kl_loss_coef: 0.001 -> 0.01 (增强KL约束，防止格式崩溃)
#   - Patient Model: Qwen3-1.7B -> Qwen3-8B (更强的对话能力)
#   - SIG LLM URL: 使用单独部署的服务
#   - use_strict_format_check: false -> true (启用格式惩罚)
#   - 添加format_penalty配置
#
# 使用方法：
#   # 默认配置（启用SIG + 格式惩罚）
#   ./20260223_run_GRPO_interactive_diagnosis_unified_sft_v4.sh
#
#   # 禁用SIG（仅使用格式惩罚和长度奖励）
#   ./20260223_run_GRPO_interactive_diagnosis_unified_sft_v4.sh --no-sig
#
# ============================================================================

set -xeuo pipefail

# ============================================================================
# 解析命令行参数
# ============================================================================
USE_STRICT_FORMAT_CHECK=true   # v4: 启用严格格式检查
USE_LENGTH_REWARD=true
LENGTH_REWARD_WEIGHT=0.2
LENGTH_MIN_TURNS=10
LENGTH_OPTIMAL_START=15
LENGTH_OPTIMAL_END=25
LENGTH_MAX_TURNS=50
CUSTOM_SUFFIX="_v4_use_token_level_reward"

# SIG (Shapley Information Gain) 奖励参数
# 注意：SIG奖励需要patient_info字段，但当前数据集中没有此字段
# 因此v4暂时禁用SIG，等数据集更新后再启用
# 如需启用SIG，请确保：
#   1. 数据集的extra_info中包含patient_info字段
#   2. SIG_LLM_BASE_URL指向单独部署的LLM服务
USE_SIG_REWARD=true
SIG_REWARD_WEIGHT=0.3
SIG_CORRECTNESS_BONUS_WEIGHT=0.2
SIG_MONTE_CARLO_K=30

# ============================================================================
# SIG 高级配置参数 (对齐 ProMed)
# ============================================================================
# Shapley 归一化方法: "softmax" 或 "z_score"
SIG_NORMALIZE_METHOD="softmax"
# Softmax 温度参数 (ProMed默认2.0, 值越大分布越平滑)
SIG_TEMPERATURE=2.0
# 字符串匹配 fallback (当LLM失败时使用)
SIG_USE_STRING_MATCHING_FALLBACK=true
SIG_STRING_MATCH_THRESHOLD=0.5
# 诊断匹配模式: "strict"(精确匹配F32.1) 或 "soft"(大类匹配F32)
SIG_DIAGNOSIS_MATCH_MODE="soft"
# Token级别奖励 (对齐 ProMed)
SIG_USE_TOKEN_LEVEL_REWARD=true
SIG_TOKEN_REWARD_ALPHA=2.0     # 问题Shapley奖励权重
SIG_TOKEN_REWARD_BETA=1.0      # 问题结果奖励权重
SIG_TOKEN_REWARD_GAMMA=3.0     # 答案正确性奖励权重
SIG_FORMAT_REWARD_WEIGHT=1.0   # 格式奖励权重

# v4: 新增格式惩罚配置
FORMAT_PENALTY_WEIGHT=0.3      # 格式错误的惩罚权重

# v4: SIG使用部署的LLM服务（与Patient Agent共用）
SIG_LLM_BASE_URL="http://192.168.5.189:8001/v1" 
# Patient Agent 配置
PATIENT_AGENT_URL="http://192.168.5.189:8011"
PATIENT_MODEL="Qwen3-8B"

# ============================================================================
# 环境配置
# ============================================================================

export NO_PROXY='192.168.5.187,192.168.5.189'
export no_proxy='192.168.5.187,192.168.5.189'

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

MODEL_PATH="/tcci_mnt/shihao/outputs/toocall_interactive/qwen3-8B_interactive_toolcall_lora-sft_lingxidiag-16k_0207"
MODEL_BASE_NAME="qwen3-8B-toocall-sft_interactive_diagnosis${CUSTOM_SUFFIX}"
NNODES=1

# 项目配置
project_name='SMHC_Interactive_Diagnosis_MultiTurn_RL'
exp_name=grpo_${MODEL_BASE_NAME}

# 确保不连接远程Ray集群
unset RAY_ADDRESS
export RAY_DISABLE_IMPORT_WARNING=1

# ============================================================================
# 算法配置 - GRPO标准配置
# v4关键修改：增强KL约束防止格式崩溃
# ============================================================================

adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
# v4: 增加KL loss系数，从0.001增加到0.01，防止模型过度偏离导致格式崩溃
kl_loss_coef=0.01
kl_loss_type=low_var_kl

# ============================================================================
# 数据配置
# ============================================================================

max_prompt_length=3096
max_response_length=10240  # 支持多轮对话
max_num_batched_tokens=16000

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
# v4: 稍微降低temperature，减少随机性，提高格式稳定性
temperature=0.7
top_p=0.95
top_k=-1
entropy_coeff=0

# ============================================================================
# 路径配置（使用 v2 数据集）
# ============================================================================

CKPTS_DIR="/tcci_mnt/shihao/project/verl/checkpoints/${project_name}/${exp_name}"
TRAIN_FILE="$HOME/psy_r1/dataset_rl/LingxiDiag-interactive-long-empathy-v4/train.parquet"
VAL_FILE="$HOME/psy_r1/dataset_rl/LingxiDiag-interactive-long-empathy-v4/val.parquet"

# ============================================================================
# 交互式问诊参数
# ============================================================================

USE_INTERACTIVE_DIAGNOSIS=True
MAX_DIALOGUE_TURNS=50
USE_ICD_REWARD=True

# ============================================================================
# 动态生成 Tool 配置文件
# ============================================================================
# v4 关键修复：
#   1. use_strict_format_check: true（启用严格格式检查）
#   2. SIG LLM使用单独部署的服务
#   3. 添加format_penalty_weight配置
#   4. 添加动态获取patient_info的支持
# ============================================================================

TOOL_CONFIG_DIR="${HOME}/psy_r1/consultation_grpo_trainer"
TOOL_CONFIG="${TOOL_CONFIG_DIR}/patient_tool_config_runtime_v4_${NOW}.yaml"

cat > "${TOOL_CONFIG}" << EOF
# 运行时动态生成的 Tool 配置文件 (v4 - SIG修复 + 格式崩溃防护)
# 生成时间: ${NOW}
# 配置选项:
#   - tool_parser: hermes（匹配 Qwen3 SFT 的 <tool_call> 格式）
#   - use_strict_format_check: ${USE_STRICT_FORMAT_CHECK}
#   - use_length_reward: ${USE_LENGTH_REWARD}
#   - use_sig_reward: ${USE_SIG_REWARD}
#   - kl_loss_coef: ${kl_loss_coef} (增强KL约束)

tools:
  - class_name: "psy_r1.verl.tools.patient_interaction_tools.AskPatientTool"
    config:
      type: native
      patient_agent_url: "${PATIENT_AGENT_URL}"
      default_patient_version: "v3"
      default_model_name: "${PATIENT_MODEL}"
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
      # v4: 启用严格格式检查（检查 think/tool_call/diagnose 三项格式要求）
      use_strict_format_check: ${USE_STRICT_FORMAT_CHECK}
      # v4: 添加格式惩罚权重
      format_penalty_weight: ${FORMAT_PENALTY_WEIGHT}
      # 长度奖励配置
      use_length_reward: ${USE_LENGTH_REWARD}
      length_reward_weight: ${LENGTH_REWARD_WEIGHT}
      length_reward_config:
        min_turns: ${LENGTH_MIN_TURNS}
        optimal_start: ${LENGTH_OPTIMAL_START}
        optimal_end: ${LENGTH_OPTIMAL_END}
        max_turns: ${LENGTH_MAX_TURNS}
      # SIG (Shapley Information Gain) 过程奖励配置
      # v4: 使用单独部署的LLM服务
      use_sig_reward: ${USE_SIG_REWARD}
      sig_reward_config:
        sig_reward_weight: ${SIG_REWARD_WEIGHT}
        correctness_bonus_weight: ${SIG_CORRECTNESS_BONUS_WEIGHT}
        monte_carlo_k: ${SIG_MONTE_CARLO_K}
        # v4: 使用单独的SIG LLM服务（不与Patient Agent共用）
        llm_base_url: "${SIG_LLM_BASE_URL}"
        understanding_model: "Qwen3-8B"
        fact_checker_model: "Qwen3-8B"
        diagnosis_model: "Qwen3-8B"
        enable_caching: true
        cache_shapley_values: true
        log_sig_details: true
        # v4: 添加更多超时和重试配置
        timeout: 90.0
        max_concurrent_requests: 16
        # v4: 当patient_info缺失时，尝试从Patient Agent获取
        fetch_patient_info_from_agent: true
        patient_agent_url: "${PATIENT_AGENT_URL}"
        # ============================================================
        # 以下为对齐 ProMed 的高级配置参数
        # ============================================================
        # Shapley 归一化方法: "softmax" 或 "z_score"
        shapley_normalize_method: "${SIG_NORMALIZE_METHOD}"
        # Softmax 温度参数
        shapley_temperature: ${SIG_TEMPERATURE}
        # 字符串匹配 fallback 配置
        use_string_matching_fallback: ${SIG_USE_STRING_MATCHING_FALLBACK}
        string_match_threshold: ${SIG_STRING_MATCH_THRESHOLD}
        # 诊断匹配模式: "strict" 或 "soft"
        diagnosis_match_mode: "${SIG_DIAGNOSIS_MATCH_MODE}"
        # Token级别奖励配置
        use_token_level_reward: ${SIG_USE_TOKEN_LEVEL_REWARD}
        token_reward_alpha: ${SIG_TOKEN_REWARD_ALPHA}
        token_reward_beta: ${SIG_TOKEN_REWARD_BETA}
        token_reward_gamma: ${SIG_TOKEN_REWARD_GAMMA}
        format_reward_weight: ${SIG_FORMAT_REWARD_WEIGHT}
    tool_schema:
      type: "function"
      function:
        name: "do_diagnose"
        description: "提交最终 ICD-10 诊断。在 diagnosis 参数中用 <box> 标签包裹 ICD-10 代码。"
        parameters:
          type: "object"
          properties:
            diagnosis:
              type: "string"
              description: "ICD-10 诊断代码，必须用 <box> 标签包裹。例如：<box>F32.1</box> 或多个诊断用分号分隔 <box>F32.1; F41.2</box>"
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
echo "GRPO 交互式问诊训练 - v4 (SIG修复 + 格式崩溃防护)"
echo "============================================================================"
echo "v4 关键修复:"
echo "  - kl_loss_coef: 0.01 (增强KL约束，防止格式崩溃)"
echo "  - use_strict_format_check: true (启用格式惩罚)"
echo "  - format_penalty_weight: ${FORMAT_PENALTY_WEIGHT}"
echo "  - Patient Model: ${PATIENT_MODEL} (更强的对话能力)"
echo "  - SIG LLM URL: ${SIG_LLM_BASE_URL} (单独部署)"
echo "  - temperature: ${temperature} (降低随机性)"
echo ""
echo "模型: ${MODEL_PATH}"
echo "Patient Agent: ${PATIENT_AGENT_URL}"
echo "训练数据: ${TRAIN_FILE}"
echo "验证数据: ${VAL_FILE}"
echo "最大对话轮数: ${MAX_DIALOGUE_TURNS}"
echo ""
echo "功能开关:"
if [ "${USE_STRICT_FORMAT_CHECK}" = "true" ]; then
    echo "  - 严格格式检查: ENABLED (penalty_weight=${FORMAT_PENALTY_WEIGHT})"
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
    echo "    SIG LLM: ${SIG_LLM_BASE_URL}"
    echo "    Monte Carlo K: ${SIG_MONTE_CARLO_K}"
    echo "    归一化方法: ${SIG_NORMALIZE_METHOD} (temperature=${SIG_TEMPERATURE})"
    echo "    诊断匹配模式: ${SIG_DIAGNOSIS_MATCH_MODE}"
    echo "    字符串匹配Fallback: ${SIG_USE_STRING_MATCHING_FALLBACK} (threshold=${SIG_STRING_MATCH_THRESHOLD})"
    if [ "${SIG_USE_TOKEN_LEVEL_REWARD}" = "true" ]; then
        echo "    Token级别奖励: ENABLED (α=${SIG_TOKEN_REWARD_ALPHA}, β=${SIG_TOKEN_REWARD_BETA}, γ=${SIG_TOKEN_REWARD_GAMMA})"
    else
        echo "    Token级别奖励: DISABLED"
    fi
else
    echo "  - SIG过程奖励: DISABLED"
fi
echo ""
echo "KL约束配置:"
echo "  - kl_loss_coef: ${kl_loss_coef}"
echo "  - kl_loss_type: ${kl_loss_type}"
echo ""
echo "Tool 配置文件: ${TOOL_CONFIG}"
echo "实验名称: ${exp_name}"
echo "============================================================================"

# ============================================================================
# API 连通性测试
# ============================================================================
echo ""
echo "============================================================================"
echo "API 连通性测试"
echo "============================================================================"

# 定义测试函数
test_api_connectivity() {
    local url=$1
    local name=$2
    local timeout=${3:-10}

    echo -n "测试 ${name} (${url})... "

    # 使用 curl 测试连通性，设置超时
    if curl -s --connect-timeout ${timeout} --max-time ${timeout} "${url}" > /dev/null 2>&1; then
        echo "✓ 连接成功"
        return 0
    else
        # 尝试健康检查端点
        if curl -s --connect-timeout ${timeout} --max-time ${timeout} "${url}/health" > /dev/null 2>&1; then
            echo "✓ 连接成功 (健康检查)"
            return 0
        fi
        echo "✗ 连接失败"
        return 1
    fi
}

# 测试 Patient Agent API (测试实际的 /api/v1/patient/chat 端点)
echo ""
echo "1. 测试 Patient Agent API..."
PATIENT_API_ENDPOINT="${PATIENT_AGENT_URL}/api/v1/patient/chat"
echo -n "测试 Patient Agent (${PATIENT_API_ENDPOINT})... "

# 发送一个简单的测试请求
PATIENT_TEST_RESPONSE=$(curl -s -w "\n%{http_code}" --connect-timeout 10 --max-time 15 \
    -X POST "${PATIENT_API_ENDPOINT}" \
    -H "Content-Type: application/json" \
    -d '{"patient_id": "test", "messages": [{"role": "user", "content": "测试"}], "patient_version": "test", "model_name": "test"}' 2>&1)

PATIENT_HTTP_CODE=$(echo "$PATIENT_TEST_RESPONSE" | tail -n1)

# 检查HTTP状态码（200, 400, 422 都表示服务可达，只是参数问题）
if [[ "$PATIENT_HTTP_CODE" =~ ^(200|400|422|500)$ ]]; then
    echo "✓ 连接成功 (HTTP ${PATIENT_HTTP_CODE})"
elif [[ "$PATIENT_HTTP_CODE" == "404" ]]; then
    echo "✗ 端点不存在 (HTTP 404)"
    echo ""
    echo "============================================================================"
    echo "错误: Patient Agent API 端点不存在!"
    echo "URL: ${PATIENT_API_ENDPOINT}"
    echo ""
    echo "请检查:"
    echo "  1. Patient Agent 服务版本是否正确"
    echo "  2. API 端点路径是否为 /api/v1/patient/chat"
    echo "  3. 确认端口号: Patient Agent 应使用 8001, SIG LLM 使用 8002"
    echo "============================================================================"
    exit 1
else
    echo "✗ 连接失败 (HTTP ${PATIENT_HTTP_CODE})"
    echo ""
    echo "============================================================================"
    echo "错误: Patient Agent API 无法连接!"
    echo "URL: ${PATIENT_AGENT_URL}"
    echo ""
    echo "请检查:"
    echo "  1. Patient Agent 服务是否已启动"
    echo "  2. 网络连接是否正常"
    echo "  3. IP 地址和端口是否正确 (应为 8001)"
    echo "============================================================================"
    exit 1
fi

# 测试 SIG LLM API (仅当启用 SIG 时)
if [ "${USE_SIG_REWARD}" = "true" ]; then
    echo ""
    echo "2. 测试 SIG LLM API..."

    # 对于 OpenAI 兼容的 API，测试 /models 端点
    SIG_MODELS_URL="${SIG_LLM_BASE_URL}/models"
    echo -n "测试 SIG LLM (${SIG_MODELS_URL})... "

    SIG_TEST_RESPONSE=$(curl -s -w "\n%{http_code}" --connect-timeout 10 --max-time 15 \
        -X GET "${SIG_MODELS_URL}" 2>&1)

    SIG_HTTP_CODE=$(echo "$SIG_TEST_RESPONSE" | tail -n1)

    if [[ "$SIG_HTTP_CODE" == "200" ]]; then
        echo "✓ 连接成功"
    elif [[ "$SIG_HTTP_CODE" =~ ^(400|401|403|500)$ ]]; then
        # 这些错误码说明服务可达，只是有其他问题
        echo "✓ 服务可达 (HTTP ${SIG_HTTP_CODE})"
    else
        echo "✗ 连接失败 (HTTP ${SIG_HTTP_CODE})"
        echo ""
        echo "============================================================================"
        echo "错误: SIG LLM API 无法连接!"
        echo "URL: ${SIG_LLM_BASE_URL}"
        echo ""
        echo "请检查:"
        echo "  1. SIG LLM 服务是否已启动 (应在 8002 端口)"
        echo "  2. 网络连接是否正常"
        echo "  3. 确认是 OpenAI 兼容的 API 服务"
        echo ""
        echo "提示: 如果不需要 SIG 奖励，可以设置 USE_SIG_REWARD=false"
        echo "============================================================================"
        exit 1
    fi
else
    echo ""
    echo "2. SIG LLM API 测试跳过 (SIG 奖励未启用)"
fi

echo ""
echo "============================================================================"
echo "✓ 所有 API 连通性测试通过"
echo "============================================================================"
echo ""

# ============================================================================
# 启动训练
# ============================================================================
# v4关键修改:
#   - kl_loss_coef=0.01 (增强KL约束)
#   - temperature=0.7 (降低随机性)

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
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_DIALOGUE_TURNS} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_DIALOGUE_TURNS} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
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
