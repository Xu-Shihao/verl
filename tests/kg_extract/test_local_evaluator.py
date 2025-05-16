#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试本地部署的vLLM API服务的Qwen模型评估功能
"""

import os
import argparse
from dotenv import load_dotenv
import json

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from verl.utils.kg_rewards import (
    qwen_compare_extracted_graph,
    qwen_reasoning_score,
    GraphEvaluationInput,
    ReasoningEvaluationInput
)

# 测试用的图数据
TEST_GRAPH = {
    "nodes": [
        {"id": 1, "label": "特斯拉", "type": "公司"},
        {"id": 2, "label": "埃隆·马斯克", "type": "人物"},
        {"id": 3, "label": "电动汽车", "type": "产品"}
    ],
    "edges": [
        {"source": 1, "target": 3, "relation": "生产"},
        {"source": 2, "target": 1, "relation": "是CEO"}
    ]
}

# 测试用的变体图（评分应该低一些）
TEST_GRAPH_VARIANT = {
    "nodes": [
        {"id": 1, "label": "特斯拉", "type": "公司"},
        {"id": 2, "label": "埃隆·马斯克", "type": "人物"}
    ],
    "edges": [
        {"source": 2, "target": 1, "relation": "拥有"}
    ]
}

# 测试用的推理文本
TEST_REASONING = """
分析文本，我们可以确定以下信息：
1. 特斯拉是一家公司，主要生产电动汽车
2. 埃隆·马斯克是特斯拉的首席执行官（CEO）
3. 特斯拉生产电动汽车这一产品类别
"""

# 测试用的变体推理（评分应该低一些）
TEST_REASONING_VARIANT = """
文本中提到特斯拉是一家电动汽车公司，由埃隆·马斯克领导。
"""

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试本地vLLM API服务的Qwen模型评估功能")
    parser.add_argument("--model_name", type=str, default="/mnt/afs/m2/models/Qwen2.5-32B-Instruct/",
                        help="Qwen模型名称")
    parser.add_argument("--api_base", type=str, default="http://10.119.21.75:9001/v1",
                        help="vLLM API服务地址")
    args = parser.parse_args()
    
    # 加载环境变量
    load_dotenv()
    
    # 清除所有代理环境变量
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["all_proxy"] = ""
    
    # 强制使用本地模型进行评估
    os.environ["USE_LOCAL_QWEN_FOR_EVAL"] = "true"
    os.environ["LOCAL_QWEN_MODEL"] = args.model_name
    os.environ["VLLM_API_BASE"] = args.api_base
    
    print(f"使用模型 {args.model_name} 进行评估测试")
    print(f"API服务地址: {args.api_base}")
    
    # 测试图评估
    print("\n=== 测试图评估 ===")
    
    # 准备输入
    graph_input = GraphEvaluationInput(
        extracted_graph=json.dumps(TEST_GRAPH, ensure_ascii=False),
        ground_truth=json.dumps(TEST_GRAPH, ensure_ascii=False)
    )
    
    # 评估相同图（应该得到高分）
    score = qwen_compare_extracted_graph(graph_input)
    print(f"相同图评估分数: {score}")
    
    # 评估变体图（应该得到较低分数）
    variant_input = GraphEvaluationInput(
        extracted_graph=json.dumps(TEST_GRAPH_VARIANT, ensure_ascii=False),
        ground_truth=json.dumps(TEST_GRAPH, ensure_ascii=False)
    )
    score = qwen_compare_extracted_graph(variant_input)
    print(f"变体图评估分数: {score}")
    
    # 测试推理评估
    print("\n=== 测试推理评估 ===")
    
    # 准备输入
    reasoning_input = ReasoningEvaluationInput(
        reasoning=TEST_REASONING,
        ground_truth=TEST_REASONING
    )
    
    # 评估相同推理（应该得到高分）
    score = qwen_reasoning_score(reasoning_input)
    print(f"相同推理评估分数: {score}")
    
    # 评估变体推理（应该得到较低分数）
    variant_reasoning_input = ReasoningEvaluationInput(
        reasoning=TEST_REASONING_VARIANT,
        ground_truth=TEST_REASONING
    )
    score = qwen_reasoning_score(variant_reasoning_input)
    print(f"变体推理评估分数: {score}")
    
    print("\n测试完成!")

if __name__ == "__main__":
    main() 