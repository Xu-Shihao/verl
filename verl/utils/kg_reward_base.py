import os
from typing import Optional, Type
from pydantic import BaseModel, Field

# 导入PydanticBaseModel，用于类型注解
from pydantic import BaseModel as PydanticBaseModel

# Pydantic模型定义
class GraphEvaluationInput(BaseModel):
    """图评估输入"""
    extracted_graph: str = Field(..., description="提取的图（JSON字符串）")
    ground_truth: str = Field(..., description="真实图（JSON字符串）")

class GraphEvaluationOutput(BaseModel):
    """图评估输出"""
    score: float = Field(..., description="评估分数（0-2范围）")
    # feedback: Optional[str] = Field(None, description="可选的评估反馈")

class ReasoningEvaluationInput(BaseModel):
    """推理评估输入"""
    reasoning: str = Field(..., description="要评估的推理")
    ground_truth: str = Field(..., description="真实推理参考")

class ReasoningEvaluationOutput(BaseModel):
    """推理评估输出"""
    score: float = Field(..., description="评估分数（0-1范围）")
    # feedback: Optional[str] = Field(None, description="可选的评估反馈")

# 评估提示模板

def get_remote_graph_eval_prompt(input_data: GraphEvaluationInput) -> str:
    """生成远程图评估提示模板"""
    return (
        "You are evaluating knowledge graph extraction results. Compare the extracted graph with the ground truth graph.\n\n"
        "Scoring Guidelines:\n"
        "2.0 - Perfect or near-perfect match with valid structure\n"
        "1.5 - Good match with valid structure and reasonable relationships\n"
        "1.0 - Valid structure but some relationship issues\n"
        "0.5 - Valid nodes but relationship issues\n"
        "0.0 - Invalid structure or completely wrong\n\n"

        "Key Points to Check:\n"
        "1. Is the JSON structure valid? (nodes and edges arrays)\n"
        "2. Do nodes have valid id, label, and type?\n"
        "3. Do edges have valid source, target, and relation?\n"
        "4. Are the relationships meaningful?\n\n"

        f"Extracted Graph (evaluate this):\n{input_data.extracted_graph}\n\n"
        f"Ground Truth Graph:\n{input_data.ground_truth}\n\n"

        "Output ONLY a number (2.0, 1.5, 1.0, 0.5, or 0.0) based on the quality of the extracted graph."
    )

def get_qwen_graph_eval_prompt(input_data: GraphEvaluationInput) -> str:
    """生成Qwen图评估提示模板"""
    return (
        "你是一名知识图谱评估专家。请对比提取的图和标准图，给出评分。\n\n"
        "评分标准:\n"
        "2.0 - 完美或几乎完美匹配，结构有效\n"
        "1.5 - 良好匹配，结构有效且关系合理\n"
        "1.0 - 结构有效但关系有一些问题\n"
        "0.5 - 节点有效但关系有问题\n"
        "0.0 - 结构无效或完全错误\n\n"

        "评估要点：\n"
        "1. JSON结构是否有效？（包含nodes和edges数组）\n"
        "2. 节点是否有有效的id、label和type？\n"
        "3. 边是否有有效的source、target和relation？\n"
        "4. 关系是否有意义？\n\n"

        f"提取的图（需要评估）:\n{input_data.extracted_graph}\n\n"
        f"标准图:\n{input_data.ground_truth}\n\n"

        "请只输出一个数字（2.0、1.5、1.0、0.5或0.0）来表示提取图的质量。不要输出其他内容。"
    )

def get_remote_reasoning_eval_prompt(input_data: ReasoningEvaluationInput) -> str:
    """生成远程推理评估提示模板"""
    return (
        "You are evaluating the quality of reasoning in knowledge graph extraction. "
        "Compare the reasoning against the ground truth and score based on these criteria:\n\n"

        "Scoring Guide:\n"
        "1.0: Excellent - Systematic analysis that correctly identifies all elements from ground truth\n"
        "0.8: Good - Correct analysis but missing minor details from ground truth\n"
        "0.5: Basic - Identifies main elements but missing significant details\n"
        "0.2: Poor - Very incomplete or vague compared to ground truth\n"
        "0.0: Invalid - Empty, irrelevant, or contradicts ground truth\n\n"

        "Ground Truth:\n" + input_data.ground_truth + "\n\n"

        "Reasoning to evaluate:\n" + input_data.reasoning + "\n\n"

        "Output only a single number (1.0, 0.8, 0.5, 0.2, or 0.0) based on how well the reasoning matches the ground truth."
    )

def get_qwen_reasoning_eval_prompt(input_data: ReasoningEvaluationInput) -> str:
    """生成Qwen推理评估提示模板"""
    return (
        "你是一名知识图谱推理评估专家。请对比待评估的推理与标准推理，给出评分。\n\n"
        "评分标准:\n"
        "1.0: 优秀 - 系统性分析，正确识别标准推理中的所有要素\n"
        "0.8: 良好 - 分析正确，但缺少标准推理中的一些细节\n"
        "0.5: 基础 - 识别主要元素，但缺少重要细节\n"
        "0.2: 较差 - 非常不完整或模糊\n"
        "0.0: 无效 - 空白、不相关或与标准推理矛盾\n\n"

        f"标准推理:\n{input_data.ground_truth}\n\n"

        f"待评估推理:\n{input_data.reasoning}\n\n"

        "请只输出一个数字（1.0、0.8、0.5、0.2或0.0）表示推理与标准推理的匹配程度。不要输出其他内容。"
    )

# 模型选择函数
def get_model_name():
    """获取当前配置的模型名称"""
    return os.getenv("LOCAL_QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct") 