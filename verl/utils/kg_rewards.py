import re
import json
import os
from typing import List, Dict, Any, Union, Optional, Type
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel

# 导入基础模型和函数
from verl.utils.kg_reward_base import (
    GraphEvaluationInput, GraphEvaluationOutput,
    ReasoningEvaluationInput, ReasoningEvaluationOutput,
    get_remote_graph_eval_prompt, get_qwen_graph_eval_prompt,
    get_remote_reasoning_eval_prompt, get_qwen_reasoning_eval_prompt,
    get_model_name
)

import dotenv
dotenv.load_dotenv()

os.environ["USE_LOCAL_QWEN_FOR_EVAL"] = "true"
os.environ["LOCAL_QWEN_MODEL"] = "/mnt/afs/m2/models/Qwen2.5-72B-Instruct/"
os.environ["VLLM_API_BASE"] = "http://10.119.16.246:9001/v1"


# 辅助函数
def extract_tag_content(text: str, tag: str) -> str:
    """从给定文本中提取被<tag>...</tag>标签包围的内容"""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    
    if not match:
        print(f"[debug] 未找到{tag}标签")
        
    return match.group(1).strip() if match else ""

def extract_json_from_answer(answer_text: str) -> str:
    """从<answer>...</answer>标签中提取JSON内容"""
    answer_content = extract_tag_content(answer_text, "answer")
    if answer_content:
        try:
            # 仅验证JSON是否可解析
            json.loads(answer_content)
            return answer_content
        except json.JSONDecodeError:
            print(f"[debug] JSON解析失败")
            return ""
    if not answer_content:
        print(f"[debug] 未找到answer标签")
    return ""

def extract_thinking_content(text: str) -> str:
    """从<think>...</think>或<think>...</think>标签中提取思考内容"""
    # 先尝试提取<think>标签内容
    thinking = extract_tag_content(text, "think")
    return thinking

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def call_llm_with_retry(client, messages, model="gpt-4o-mini", **kwargs):
    """带有重试逻辑的LLM API调用包装器"""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        return completion
    except Exception as e:
        print(f"[debug] LLM API call failed: {str(e)}")
        raise

def call_vllm_qwen(prompt: str, model_name: str = None, max_tokens: int = 256) -> str:
    """
    调用本地已部署的vllm Qwen模型服务（普通文本响应方式）
    """
    # 如果未指定模型名称，使用配置的默认值
    if model_name is None:
        model_name = get_model_name()
        
    try:
        from openai import OpenAI
        
        # 从环境变量获取API基础URL，默认为本地vLLM服务
        api_base = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
        
        # 创建OpenAI客户端连接到vLLM服务
        client = OpenAI(
            api_key="EMPTY",  # vLLM服务通常不需要API密钥
            base_url=api_base,
        )
        
        # 调用聊天接口
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # 低温度使输出更确定性
            top_p=0.9,
            max_tokens=max_tokens,
            extra_body={
                "repetition_penalty": 1.05,
            },
        )
        
        # 提取生成的文本
        generated_text = chat_response.choices[0].message.content
        
        return generated_text
    except Exception as e:
        print(f"[debug] vLLM调用失败: {str(e)}")
        return ""

def call_vllm_qwen_with_schema(prompt: str, response_schema: Type[BaseModel], 
                              model_name: str = None, 
                              max_tokens: int = 256,
                              system_prompt: str = "") -> Optional[BaseModel]:
    """
    调用本地已部署的vllm模型服务，使用结构化响应格式
    """
    # 如果未指定模型名称，使用配置的默认值
    if model_name is None:
        model_name = get_model_name()
    
    try:
        from openai import OpenAI
        
        # 从环境变量获取API基础URL，默认为本地vLLM服务
        api_base = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
        
        # 创建OpenAI客户端连接到vLLM服务
        client = OpenAI(
            api_key="EMPTY",  # vLLM服务通常不需要API密钥
            base_url=api_base,
        )
        
        # 准备消息
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # 调用parse接口进行结构化输出
        completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=messages,
            response_format=response_schema,
            temperature=0.1,  # 低温度使输出更确定性
            top_p=0.9,
            max_tokens=max_tokens,
            extra_body={
                "repetition_penalty": 1.05,
            },
        )
        
        # 提取解析后的结构化数据
        message = completion.choices[0].message
        if hasattr(message, 'parsed') and message.parsed:
            return message.parsed
        else:
            print("[debug] 无法获取解析后的结构化数据")
            return None
            
    except Exception as e:
        print(f"[debug] 结构化vLLM调用失败: {str(e)}")
        return None

# 格式检查奖励函数
def format_reward(solution_str: str, **kwargs) -> Dict[str, Any]:
    """检查输出是否符合特定格式的奖励函数"""
    # 检查<think>...</think>或<think>...</think>标签
    has_thinking = bool(extract_thinking_content(solution_str))
    
    # 检查<answer>...</answer>标签
    has_answer = bool(extract_tag_content(solution_str, "answer"))
    
    # 计算分数 (0.5分)
    score = 0.5 if (has_thinking and has_answer) else 0.0
    
    return {
        "score": score,
        "format_score": score,
        "has_thinking": has_thinking,
        "has_answer": has_answer
    }

def xml_count_reward(solution_str: str, **kwargs) -> Dict[str, Any]:
    """基于XML标签计数的奖励函数"""
    count = 0.0
    
    # 检查reasoning标签
    if solution_str.count("<think>\n") == 1:
        count += 0.125
    if solution_str.count("\n</think>\n") == 1:
        count += 0.125
    
    # 检查answer标签
    if solution_str.count("\n<answer>\n") == 1:
        count += 0.125
    if solution_str.count("\n</answer>") == 1:
        count += 0.125
    
    return {
        "score": count,
        "xml_count": count
    }

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def strict_format_reward(solution_str: str, **kwargs) -> Dict[str, Any]:
    """检查输出是否符合严格的XML格式要求"""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    match = re.match(pattern, solution_str, re.DOTALL)
    score = 0.5 if match else 0.0
    
    return {
        "score": score,
        "strict_format_score": score
    }

def soft_format_reward(solution_str: str, **kwargs) -> Dict[str, Any]:
    """检查输出是否符合宽松的XML格式要求"""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    match = re.match(pattern, solution_str, re.DOTALL)
    score = 0.5 if match else 0.0
    
    return {
        "score": score,
        "soft_format_score": score
    }

# 图正确性奖励函数
def basic_graph_score(extracted_graph: str, ground_truth: str) -> float:
    """基于简单规则对图进行评分"""
    try:
        # 检查JSON有效性
        comp_json = json.loads(extracted_graph)
        truth_json = json.loads(ground_truth)
        
        # 提取基本信息
        comp_nodes = comp_json.get("nodes", [])
        comp_edges = comp_json.get("edges", [])
        truth_nodes = truth_json.get("nodes", [])
        truth_edges = truth_json.get("edges", [])
        
        # 节点数量比较
        if len(comp_nodes) == 0:
            return 0.0
        
        node_ratio = min(len(comp_nodes) / max(1, len(truth_nodes)), 1.0)
        
        # 边数量比较
        if len(truth_edges) == 0:
            edge_ratio = 1.0 if len(comp_edges) == 0 else 0.5
        else:
            edge_ratio = min(len(comp_edges) / max(1, len(truth_edges)), 1.0)
        
        # 计算边属性覆盖率
        edge_attr_score = 0.0
        required_attrs = ["source", "target", "relation", "description", "keywords", "strength", "msg_ids"]
        
        if comp_edges:
            # 检查第一条边是否具有所有必要属性
            first_edge = comp_edges[0]
            attrs_present = sum(1 for attr in required_attrs if attr in first_edge)
            edge_attr_score = attrs_present / len(required_attrs)
        
        # 计算最终分数 (满分2.0)
        score = (node_ratio * 0.4 + edge_ratio * 0.4 + edge_attr_score * 0.2) * 2.0
        return min(score, 2.0)
    
    except Exception as e:
        # JSON解析错误或其他问题
        return 0.0

def llm_compare_extracted_graph(extracted_graph: str, ground_truth: str) -> float:
    """
    使用LLM比较提取的图和真实图
    """
    try:
        # 首先验证JSON
        comp_json = json.loads(extracted_graph)
        truth_json = json.loads(ground_truth)
    except json.JSONDecodeError:
        print(f"[debug] JSON validation failed")
        return 0.0
    
    # 创建Pydantic模型实例
    input_data = GraphEvaluationInput(
        extracted_graph=extracted_graph,
        ground_truth=ground_truth
    )

    # 选择使用哪种评估方式
    # use_local_model = os.getenv("USE_LOCAL_QWEN_FOR_EVAL", "false").lower() == "true"
    use_local_model = True
    
    if use_local_model:
        return qwen_compare_extracted_graph(input_data)
    else:
        return remote_llm_compare_extracted_graph(input_data)

def remote_llm_compare_extracted_graph(input_data: GraphEvaluationInput) -> float:
    """使用远程OpenAI/Azure模型评估抽取的图"""
    prompt = get_remote_graph_eval_prompt(input_data)

    try:
        if os.getenv("AZURE_OPENAI_API_KEY"):
            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-02-15-preview",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT") or "",
            )
            model = "gpt-4o-mini"  # 或你的Azure部署名称
        else:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model = "gpt-4o-mini"

        completion = call_llm_with_retry(
            client,
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.1,
            max_tokens=10
        )

        score_text = completion.choices[0].message.content.strip()
        
        # 更稳健的分数解析
        score_match = re.findall(r"[0-2]\.?[05]?", score_text)
        if score_match:
            score = float(score_match[0])
            return score
        else:
            return 0.0

    except Exception as e:
        print(f"[debug] LLM call failed with error: {str(e)}")
        # 添加基本的回退评分
        return basic_graph_score(input_data.extracted_graph, input_data.ground_truth)

def qwen_compare_extracted_graph(input_data: GraphEvaluationInput) -> float:
    """使用本地vllm部署的Qwen模型评估抽取的图"""
    model_name = get_model_name()
    prompt = get_qwen_graph_eval_prompt(input_data)
    
    try:
        # 尝试使用结构化输出方式
        result = call_vllm_qwen_with_schema(
            prompt=prompt,
            response_schema=GraphEvaluationOutput,
            model_name=model_name,
            max_tokens=256
        )
        
        if result and hasattr(result, 'score'):
            score = result.score
            return score
            
        # 如果结构化解析失败，退回到普通文本方式
        response = call_vllm_qwen(prompt, model_name=model_name, max_tokens=16)
        
        # 解析分数
        score_match = re.findall(r"[0-2]\.?[05]?", response)
        if score_match:
            score = float(score_match[0])
            return score
        else:
            # 回退到基本评分
            return basic_graph_score(input_data.extracted_graph, input_data.ground_truth)
            
    except Exception as e:
        print(f"[debug] Qwen evaluation failed with error: {str(e)}")
        # 回退到基本评分
        return basic_graph_score(input_data.extracted_graph, input_data.ground_truth)

def graph_correctness_reward(solution_str: str, ground_truth_answer: str, **kwargs) -> Dict[str, Any]:
    """评估抽取的图与标准图的相似度，使用LLM进行评估"""
    # 从回答中提取JSON
    extracted_json = extract_tag_content(solution_str, "answer")
    
    # 如果ground_truth_answer中包含<answer>和</answer>标签，则提取标签内容
    if "<answer>" in ground_truth_answer and "</answer>" in ground_truth_answer:
        ground_truth_answer = extract_tag_content(ground_truth_answer, "answer") 
    
    # 如果提取失败，返回0分
    if not extracted_json:
        return {
            "score": 0.0,
            "graph_score": 0.0,
            "valid_json": False
        }
    
    # 验证JSON格式
    try:
        json.loads(extracted_json)
        is_valid_json = True
    except json.JSONDecodeError:
        return {
            "score": 0.0,
            "graph_score": 0.0,
            "valid_json": False
        }
    
    # 计算图相似度分数 - 优先使用LLM评估
    use_llm = os.getenv("USE_LLM_FOR_GRAPH_EVAL", "true").lower() == "true"
    
    if use_llm and is_valid_json:
        score = llm_compare_extracted_graph(extracted_json, ground_truth_answer)
    else:
        score = basic_graph_score(extracted_json, ground_truth_answer) if is_valid_json else 0.0
    
    return {
        "score": score,
        "graph_score": score,
        "valid_json": is_valid_json
    }

def llm_reasoning_score(reasoning: str, ground_truth: str) -> float:
    """使用LLM评估推理质量"""
    # 创建Pydantic模型实例
    input_data = ReasoningEvaluationInput(
        reasoning=reasoning,
        ground_truth=ground_truth
    )
    
    # 选择使用哪种评估方式
    # use_local_model = os.getenv("USE_LOCAL_QWEN_FOR_EVAL", "false").lower() == "true"
    use_local_model = True
    
    if use_local_model:
        return qwen_reasoning_score(input_data)
    else:
        return remote_llm_reasoning_score(input_data)

def remote_llm_reasoning_score(input_data: ReasoningEvaluationInput) -> float:
    """使用远程OpenAI/Azure模型评估推理质量"""
    prompt = get_remote_reasoning_eval_prompt(input_data)
    
    try:
        if os.getenv("AZURE_OPENAI_API_KEY"):
            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-02-15-preview",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT") or "",
            )
        else:
            from openai import OpenAI
            client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )

        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # 根据需要调整
            messages=[{"role": "user", "content": prompt}],
        )
        score_text = completion.choices[0].message.content
        if score_text:
            score_text = score_text.strip()
            match = re.search(r"([\d\.]+)", score_text)
            if match:
                return float(match.group(1))
        return 0.0  # 如果响应中没有找到有效分数则返回0.0
    except Exception as e:
        print("Remote LLM call for reasoning evaluation failed:", e)
        return 0.0

def qwen_reasoning_score(input_data: ReasoningEvaluationInput) -> float:
    """使用本地vllm部署的Qwen模型评估推理质量"""
    model_name = get_model_name()
    prompt = get_qwen_reasoning_eval_prompt(input_data)
    
    try:
        # 尝试使用结构化输出方式
        result = call_vllm_qwen_with_schema(
            prompt=prompt,
            response_schema=ReasoningEvaluationOutput,
            model_name=model_name,
            max_tokens=256
        )
        
        if result and hasattr(result, 'score'):
            score = result.score
            return score
            
        # 如果结构化解析失败，退回到普通文本方式
        response = call_vllm_qwen(prompt, model_name=model_name, max_tokens=16)
        
        # 解析分数
        score_match = re.search(r"(0\.[0258]|1\.0|1)", response)
        if score_match:
            score = float(score_match.group(1))
            return score
        else:
            print(f"[debug] Qwen reasoning evaluation failed with error: {response}")
            return 0.0  # 没有找到有效分数
            
    except Exception as e:
        print(f"[debug] Qwen reasoning evaluation failed with error: {str(e)}")
        return 0.0

def reasoning_quality_reward(solution_str: str, ground_truth_reasoning: str, **kwargs) -> Dict[str, Any]:
    """评估推理的质量，使用LLM进行评估"""
    # 提取推理部分
    reasoning = extract_thinking_content(solution_str)
    
    if not reasoning:
        return {
            "score": 0.0,
            "reasoning_score": 0.0,
            "has_reasoning": False
        }
    
    # 使用LLM评估推理质量
    use_llm = os.getenv("USE_LLM_FOR_REASONING_EVAL", "true").lower() == "true"
    
    if use_llm and ground_truth_reasoning and len(ground_truth_reasoning) > 0:
        total_score = llm_reasoning_score(reasoning, ground_truth_reasoning)
    else:
        # 回退到基础评分方法
        min_length = 50  # 最小有效长度
        ideal_length = 200  # 理想长度
        
        if len(reasoning) < min_length:
            base_score = 0.2
        else:
            base_score = min(0.8, len(reasoning) / ideal_length * 0.8)
        
        # 关键词检查 - 图相关术语
        graph_keywords = ["node", "edge", "entity", "relationship", "graph"]
        graph_keyword_bonus = sum(1 for kw in graph_keywords if kw.lower() in reasoning.lower()) / len(graph_keywords) * 0.1
        
        # 关键词检查 - 英文专业术语
        domain_keywords = ["relation", "description", "keywords", "strength", "msg_ids"]
        domain_keyword_bonus = sum(1 for kw in domain_keywords if kw.lower() in reasoning.lower()) / len(domain_keywords) * 0.1
        
        # 如果提供了ground_truth_reasoning，可以进行更深入的比较
        comparison_bonus = 0.0
        if ground_truth_reasoning and len(ground_truth_reasoning) > 20:
            # 简单的文本重叠率作为相似度度量
            # 在实际应用中可以使用更复杂的语义相似度算法
            gt_reasoning = ground_truth_reasoning.lower()
            user_reasoning = reasoning.lower()
            
            # 计算词汇重叠率
            gt_words = set(gt_reasoning.split())
            user_words = set(user_reasoning.split())
            if gt_words:
                overlap = len(gt_words.intersection(user_words)) / len(gt_words)
                comparison_bonus = overlap * 0.2
        
        total_score = min(1.0, base_score + graph_keyword_bonus + domain_keyword_bonus + comparison_bonus)
    
    return {
        "score": total_score,
        "reasoning_score": total_score,
        "has_reasoning": True,
        "reasoning_length": len(reasoning)
    }

# 综合奖励函数
def kg_extraction_reward(data_source: str, solution_str: str, ground_truth: dict, extra_info=None) -> Dict[str, Any]:
    """综合奖励函数，整合多个奖励指标"""
    # 获取ground_truth中的答案和推理部分
    ground_truth_answer = ground_truth.get("ground_truth_answer", "{}")
    ground_truth_reasoning = ground_truth.get("ground_truth_reasoning", "")
    
    
    os.environ["USE_LOCAL_QWEN_FOR_EVAL"] = "true"
    os.environ["LOCAL_QWEN_MODEL"] = "/mnt/tanka/models/Qwen2.5-32B-Instruct"
    os.environ["VLLM_API_BASE"] = "http://10.119.21.75:9001/v1"

    # 1. 格式奖励
    format_reward_result = format_reward(solution_str)
    
    # 2. XML标签奖励
    xml_reward_result = xml_count_reward(solution_str)
    
    # 3. 严格格式奖励
    strict_format_result = strict_format_reward(solution_str)
    
    # 4. 宽松格式奖励
    soft_format_result = soft_format_reward(solution_str)
    
    # 5. 图正确性奖励
    graph_reward_result = graph_correctness_reward(solution_str, ground_truth_answer)
    
    # 6. 推理质量奖励
    reasoning_reward_result = reasoning_quality_reward(solution_str, ground_truth_reasoning)
    
    # 计算加权总分
    format_weight = 0.05
    xml_weight = 0.05
    strict_format_weight = 0.05
    soft_format_weight = 0.05
    graph_weight = 0.6
    reasoning_weight = 0.2
    
    total_score = (
        format_reward_result["score"] * format_weight +
        xml_reward_result["score"] * xml_weight +
        strict_format_result["score"] * strict_format_weight +
        soft_format_result["score"] * soft_format_weight +
        graph_reward_result["score"] * graph_weight +
        reasoning_reward_result["score"] * reasoning_weight
    )
    
    return {
        "score": total_score,
        "format_score": format_reward_result["score"],
        "xml_count": xml_reward_result["score"],
        "strict_format_score": strict_format_result["score"],
        "soft_format_score": soft_format_result["score"],
        "graph_score": graph_reward_result["score"],
        "reasoning_score": reasoning_reward_result["score"]
    }

# 主函数：用于verl框架
def compute_kg_extraction_reward(data_source, solution_str, ground_truth, extra_info=None):
    """知识图谱抽取奖励计算主函数"""
    # 如果ground_truth为None或为空，创建一个空的默认值
    if ground_truth is None:
        ground_truth = {"ground_truth_answer": "{}", "ground_truth_reasoning": ""}
    return kg_extraction_reward(data_source, solution_str, ground_truth, extra_info) 

if __name__ == "__main__":
    
    text = """<think>
The text contains information about a company's public holiday schedule for the first quarter of 2025. The key entities are Kim YEO (the sender), the company (the organization), and the public holiday schedule (the document). The main relationships are between the sender and the company, and between the company and the public holiday schedule.

- Kim YEO is the sender of the message and the document.
- The company is the recipient of the message and the owner of the public holiday schedule.
- The public holiday schedule is a document that contains the information about the holidays.

The relationships can be described as follows:
1. Kim YEO (sender) -> Company (recipient): Kim YEO is sending the message and the document to the company.
2. Company (recipient) -> Public Holiday Schedule (document): The company is receiving and distributing the public holiday schedule.
</think>

<answer>
{
  "nodes": [
    {"id": 1, "label": "Kim YEO", "type": "Person"},
    {"id": 2, "label": "Company", "type": "Organization"},
    {"id": 3, "label": "Public Holiday Schedule", "type": "Document"}
  ],
  "edges": [
    {
      "source": 1,
      "target": 2,
      "relation": "Sends",
      "description": "Kim YEO sends the message and the document to the company.",
      "keywords": ["Kim YEO", "Company", "Message", "Document"],
      "strength": 8,
      "msg_ids": [0, 1]
    },
    {
      "source": 2,
      "target": 3,
      "relation": "Receives",
      "description": "The company receives and distributes the public holiday schedule.",
      "keywords": ["Company", "Public Holiday Schedule", "Distributes"],
      "strength": 8,
      "msg_ids": [0, 1]
    }
  ]
}
</answer>"""
    print(extract_json_from_answer(text))
    print(extract_thinking_content(text))
    