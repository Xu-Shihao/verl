#!/bin/bash
# =============================================================================
# Patient Agent 启动脚本
# 1. 启动 vLLM 部署 Qwen3-1.7B 模型
# 2. 启动 Patient API 服务（支持多版本 Patient: v1, mdd5k, v3, cot）
# =============================================================================

set -e

# ===== 配置参数（按需修改）=====

# vLLM 模型配置
MODEL_PATH="/tcci_mnt/shihao/models/Qwen3-8B"
SERVED_MODEL_NAME="Qwen3-8B"
VLLM_PORT=8001
CUDA_DEVICES="0,1,2,3,4,5,6,7"

# Patient API 配置
PATIENT_API_PORT=8011
PATIENT_DATA_FILE="/tcci_mnt/shihao/project/Lingxi_annotation_0111/raw_data/LingxiDiag-16K_train_data.json"
PATIENT_VERSION="v3"  # 可选: v1, mdd5k, v3, cot

# 日志目录
LOG_DIR="/tcci_mnt/shihao/logs"

# 获取本机IP地址
HOST_IP=$(hostname -I | awk '{print $1}')

# =============================================================================

mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "1. 启动 vLLM 服务 (${SERVED_MODEL_NAME})"
echo "   模型: ${MODEL_PATH}"
echo "   端口: ${VLLM_PORT}, GPU: ${CUDA_DEVICES}"
echo "============================================================"

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} nohup python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --port ${VLLM_PORT} \
    --host 0.0.0.0 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 16000 \
    --dtype bfloat16 \
    > "${LOG_DIR}/vllm_patient_${VLLM_PORT}.log" 2>&1 &

echo "vLLM PID: $!, 日志: ${LOG_DIR}/vllm_patient_${VLLM_PORT}.log"

# 等待 vLLM 就绪
echo "等待 vLLM 服务就绪..."
for i in {1..300}; do
    if curl -s "http://${HOST_IP}:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM 服务已就绪!"
        break
    fi
    [ $i -eq 300 ] && echo "警告: vLLM 启动超时"
    sleep 2
done

echo ""
echo "============================================================"
echo "2. 启动 Patient API 服务"
echo "   端口: ${PATIENT_API_PORT}, Patient版本: ${PATIENT_VERSION}"
echo "============================================================"

export PATIENT_USE_OPENROUTER=false
export OFFLINE_PATIENT_MODEL="${SERVED_MODEL_NAME}"
export OFFLINE_PATIENT_PORTS="${VLLM_PORT}"
export VLLM_PATIENT_IP=""
export PATIENT_DATA_FILE="${PATIENT_DATA_FILE}"
export PATIENT_VERSION="${PATIENT_VERSION}"

cd /tcci_mnt/shihao/project/Lingxi_annotation_0210

nohup python3 src/patient/patient_api.py \
    --host 0.0.0.0 \
    --port ${PATIENT_API_PORT} \
    --data-file "${PATIENT_DATA_FILE}" \
    > "${LOG_DIR}/patient_api_${PATIENT_API_PORT}.log" 2>&1 &

echo "Patient API PID: $!, 日志: ${LOG_DIR}/patient_api_${PATIENT_API_PORT}.log"

echo ""
echo "============================================================"
echo "服务启动完成"
echo "============================================================"
echo "vLLM: http://${HOST_IP}:${VLLM_PORT}/v1"
echo "Patient API: http://${HOST_IP}:${PATIENT_API_PORT}"
echo ""
echo "测试请求:"
echo "curl -X POST 'http://${HOST_IP}:${PATIENT_API_PORT}/api/v1/patient/chat' -H 'Content-Type: application/json' -d '{\"patient_id\": \"300005853\", \"messages\": [{\"role\": \"user\", \"content\": \"你好\"}], \"patient_version\": \"${PATIENT_VERSION}\", \"model_name\": \"${SERVED_MODEL_NAME}\"}'"
