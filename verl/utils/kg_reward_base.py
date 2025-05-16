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
        "你是一名知识图谱评估专家。请对比提取的图和标准图，严格评估并给出评分。\n\n"
        "评分标准:\n"
        "2.0 - 完美匹配：所有节点和关系与标准图完全一致，结构完全正确\n"
        "1.5 - 良好匹配：包含标准图中绝大部分节点（≥90%），关系基本正确，结构有效\n"
        "1.0 - 一般匹配：包含标准图中大部分节点（≥70%），部分关系有问题，但整体结构有效\n"
        "0.5 - 较差匹配：包含标准图中部分节点（≥50%），多数关系有问题或结构不完整\n"
        "0.0 - 无效匹配：节点缺失严重（<50%），关系混乱或结构完全错误\n\n"

        "评估要点（严格按以下顺序评估）：\n"
        "1. 节点完整性：提取的图是否包含标准图中的所有关键节点？缺失节点比例是多少？\n"
        "2. JSON结构有效性：是否正确包含nodes和edges数组，格式是否规范？\n"
        "3. 节点属性完整性：节点是否都有正确的id、label和type？\n"
        "4. 边的正确性：是否正确连接了应有的节点（source和target），relation是否准确？\n"
        "5. 关系完整性：是否包含标准图中的所有关键关系？缺失比例是多少？\n\n"

        f"提取的图（需要评估）:\n{input_data.extracted_graph}\n\n"
        f"标准图:\n{input_data.ground_truth}\n\n"

        "请只输出一个数字（2.0、1.5、1.0、0.5或0.0）来表示提取图的质量。必须严格按照上述标准评分，不要输出其他内容。"
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
        "你是一名知识图谱推理评估专家。请对比待评估的推理与标准推理，严格评估并给出评分。请注意，你的评分标准必须严格，不要过于宽松。\n\n"
        "评分标准:\n"
        "1.0: 优秀 - 完全正确且全面的分析，100%准确识别标准推理中的所有关键要素和逻辑关系，无任何错误或遗漏\n"
        "0.8: 良好 - 分析大体正确（≥90%准确），包含大部分关键要素，但有少量细节遗漏或轻微不准确\n"
        "0.5: 一般 - 基本理解（约70%准确），识别了部分关键要素，但缺少多项重要细节或存在明显错误\n"
        "0.2: 较差 - 理解有限（约40%准确），大量关键要素缺失，推理过程不完整或存在严重错误\n"
        "0.0: 无效 - 完全不相关或错误（<20%准确），与标准推理矛盾，或根本没有有效推理\n\n"

        "评估要点（必须全面检查以下所有方面）：\n"
        "1. 推理完整性：是否包含标准推理中的所有关键步骤和结论？\n"
        "2. 逻辑准确性：推理过程是否符合逻辑，各步骤之间的关联是否正确？\n"
        "3. 要素识别：是否正确识别了标准推理中的所有实体、关系和属性？\n"
        "4. 细节精确度：细节描述是否准确，有无错误理解或错误关联？\n"
        "5. 整体一致性：整体推理框架是否与标准推理一致？\n\n"

        f"标准推理:\n{input_data.ground_truth}\n\n"

        f"待评估推理:\n{input_data.reasoning}\n\n"

        "请只输出一个数字（1.0、0.8、0.5、0.2或0.0）表示推理与标准推理的匹配程度。注意，只有在完全满足1.0分标准的情况下才能给出满分，必须严格按照评分标准执行，避免过高评分。不要输出其他内容。"
    )

# 模型选择函数
def get_model_name():
    """获取当前配置的模型名称"""
    return os.getenv("LOCAL_QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct") 