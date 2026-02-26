#!/bin/bash
# SIG LLM 部署脚本（简洁版）
# 用于 v4 训练的 SIG 过程奖励计算
# 端口 8002 与 v4 训练脚本中的 SIG_LLM_BASE_URL 匹配

# 方案 A：使用 Qwen3-32B（推荐，2 张 GPU）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -m vllm.entrypoints.openai.api_server \
    --model /tcci_mnt/shihao/models/Qwen3-32B \
    --served-model-name Qwen3-32B \
    --port 8002 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --reasoning-parser deepseek_r1 \
    --dtype bfloat16 > /tcci_mnt/shihao/logs/sig_llm_8002.log 2>&1 &

echo "SIG LLM 服务已启动"
echo "  端口: 8002"
echo "  日志: /tcci_mnt/shihao/logs/sig_llm_8002.log"
echo "  测试: curl http://localhost:8002/v1/models"

# 方案 B：使用更小的模型（单 GPU，更省资源）
# CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
#     --model /tcci_mnt/zhoutiancheng/models/qwen/qwen3-30b-a3b-instruct-2507/ \
#     --served-model-name qwen3-30b-SIG \
#     --port 8002 \
#     --host 0.0.0.0 \
#     --trust-remote-code \
#     --tensor-parallel-size 1 \
#     --gpu-memory-utilization 0.9 \
#     --max-model-len 8192 \
#     --reasoning-parser deepseek_r1 \
#     --dtype bfloat16 > /tcci_mnt/shihao/logs/sig_llm_8002.log 2>&1 &
