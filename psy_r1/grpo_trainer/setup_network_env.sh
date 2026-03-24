#!/usr/bin/env bash
# 网络环境设置脚本
# 在所有节点上运行此脚本来设置网络环境变量

# 设置网络接口
export GLOO_SOCKET_IFNAME=eth1
export NCCL_SOCKET_IFNAME=eth1
export NCCL_TIMEOUT=1800

# 将环境变量写入到系统环境文件，确保 Ray workers 能够继承
# 注意：这需要在 Ray 集群启动之前执行

echo "Network environment variables have been set:"
echo "GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"
echo "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
echo "NCCL_TIMEOUT=$NCCL_TIMEOUT"
echo ""
echo "Please ensure these environment variables are set on ALL nodes before starting the training."
