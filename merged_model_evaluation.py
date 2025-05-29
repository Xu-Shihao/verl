from vllm import LLM, SamplingParams
import sys
sys.path.append('/mnt/tanka/fengji/works/Omne-RAG')
sys.path.append('/mnt/afs/tanka/shihao/project/verl')  # 添加verl路径
import os
for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
    os.environ.pop(var, None)
import re
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from tqdm import tqdm
# 导入训练时使用的奖励函数
from verl.utils.reward_score.qa_em import extract_solution, normalize_answer

from search_r1.llm_agent.tools.edge_to_node import Knowledge_Graph_Node_Edge_To_Node_Tool
from search_r1.llm_agent.tools.entity_matcher import Knowledge_Graph_Entity_Matcher_Tool
from search_r1.llm_agent.tools.node_info import Knowledge_Graph_Get_Node_Info_Tool

# ============= 配置参数 =============
# TODO: 修改base模型和adapter路径，全参数微调设置model_name并把adapter_name设置为空，lora微调设置adapter_name和adapter_path
# model_name = ""
# model_name_or_path = "/mnt/tanka/models/Qwen3-14B" 
# adapter_name = "qwen3_14b_kg_new"
# adapter_path = f"/mnt/tanka/chenweiling/Codes/LLaMA-Factory/saves/{adapter_name}/lora/sft"

model_name = "lightrag-ppo-qwen2.5-14b-it-em-0522-step350"
model_name_or_path = f"/mnt/afs/tanka/shihao/project/verl/merged_models/{model_name}"
adapter_name = ""
adapter_path = ""

# 全局变量
results = []
processed_count = 0
output_path = None

# ============= 工具初始化 =============
def init_knowledge_graph_tools(endpoint: str):
    """初始化知识图谱工具
    
    Args:
        endpoint: 知识图谱服务的endpoint地址
    
    Returns:
        tuple: (entity_matcher, node_info, edge_to_node) 三个工具实例
    """
    entity_matcher = Knowledge_Graph_Entity_Matcher_Tool(
        api_endpoint=endpoint
    )
    
    node_info = Knowledge_Graph_Get_Node_Info_Tool(
        api_endpoint=endpoint
    )
        
    edge_to_node = Knowledge_Graph_Node_Edge_To_Node_Tool(
        api_endpoint=endpoint
    )
    
    return entity_matcher, node_info, edge_to_node

# 初始化知识图谱工具
endpoint = '10.119.16.48:9000'
entity_matcher, node_info, edge_to_node = init_knowledge_graph_tools(endpoint)

# ============= 辅助函数 =============
def save_buffer_to_file(output_path, batch_results):
    """将缓冲区中的结果保存到文件"""
    try:        
        # 如果文件存在，读取现有结果
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
        else:
            existing_results = []
        
        # 添加缓冲区中的结果
        existing_results.extend(batch_results)
        
        # 保存更新后的结果
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(existing_results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存缓冲区结果时出错: {str(e)}")

def call_tool(tool_name, node_name, graph_type_extracted):
    """调用知识图谱工具"""
    try:
        if tool_name == "node_info":
            return node_info.execute(node_name=node_name, graph_type=graph_type_extracted)
        elif tool_name == "entity_matcher":
            return entity_matcher.execute(node_name=node_name, graph_type=graph_type_extracted)
        else:
            return ""
    except Exception as e:
        return f"Error in tool execution: {str(e)}"


# ============= 核心处理函数 =============
def load_dataset(args):
    """加载数据集并返回 dataset 和 output_path"""
    global output_path
    if args.type == '12hop':
        if args.json:
            file_path = "/mnt/tanka/chenweiling/Data/kg_multihop/test_1_2_hop_json.json"
        else:
            file_path = "/mnt/tanka/han.zhao/LLaMA-Factory/data/test_1_2_hop_clean.json"
    else:  # 34hop
        if args.json:
            file_path = "/mnt/tanka/chenweiling/Data/kg_multihop/test_3_4_hop_json.json"
        else:
            file_path = "/mnt/tanka/han.zhao/LLaMA-Factory/data/test_jifeng_v2_clean.json"

    enable_lora = bool(adapter_name.strip())
    lora_flag = "lora" if enable_lora else "full"
    if adapter_name != "":
        output_path = f"results/{adapter_name}_{lora_flag}_test_{args.type}_json{args.json}_think{args.think}.json"
    else:
        output_path = f"results/{model_name}_{lora_flag}_test_{args.type}_json{args.json}_think{args.think}.json"
    print(f"文件路径: {file_path}")
    print(f"输出路径: {output_path}")

    # 如果输出文件已存在，则删除
    
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"已删除已存在的输出文件: {output_path}")

    # 读取测试集
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset, output_path

def init_model():
    """初始化vLLM引擎和LoRA"""
    print("开始初始化模型...")
    print(f"基础模型路径: {model_name_or_path}")
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    # 根据 adapter_path 是否为空来决定是否启用 LoRA
    enable_lora = bool(adapter_name.strip())
    if enable_lora:
        print(f"LoRA适配器路径: {adapter_path}")
    else:
        print("未启用LoRA")
    
    engine_args = {
        "model": model_name_or_path,
        "trust_remote_code": True,
        "dtype": "float16",
        "max_model_len": 10240,
        "tensor_parallel_size": 8, # TODO: 修改tensor_parallel_size
        "gpu_memory_utilization": 0.8,
        "disable_log_stats": True,
        # "enable_lora": enable_lora,
        # "max_lora_rank": 32,
    }
    print(f"模型参数配置: {json.dumps(engine_args, indent=2, ensure_ascii=False)}")
    
    # 只有在启用 LoRA 时才创建 LoRARequest
    lora_request = None# LoRARequest("default", 1, adapter_path) if enable_lora else None
    model = LLM(**engine_args)
    print("模型初始化完成")
    
    return model, lora_request, tokenizer

def batch_process_samples(batch_samples, model, sampling_params, lora_request, stats, tokenizer, think=True, use_json=True):
    """批量处理样本"""
    batch_results = []
    dialogs = []
    questions = []
    golden_answers = []
    graph_types = []

    for sample in batch_samples:
        question = sample.get("question", "")
        if not question:
            continue
        graph_type = sample.get("graph_type", sample.get("domain", ""))
        conversations = sample.get("conversations", [])
        last_value = conversations[-1]["value"] if conversations else ""
        golden_answer = sample.get("golden_answer", "")
        if golden_answer == "":
            match = re.search(r"<answer>(.*?)</answer>", last_value, re.DOTALL)
            golden_answer = match.group(1).strip() if match else "no_gold_answer"

        # 构建system prompt
        system_prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>"""

        # 根据think参数选择不同的user prompt
        user_prompt = f"""<|im_start|>user
Answer the given question using knowledge graph tools. 

You should use an iterative process of reasoning and tool usage to solve the problem. For each step:
1. First, think about what you know and what information you need by writing your thoughts inside <think> and </think> tags
2. Then, if needed, use an appropriate tool by putting the tool call inside <search> and </search> tags, you can only use one tool at a time
3. After receiving tool results, think again about what you've learned and what to do next
4. Repeat this process of thinking and searching until you have enough information to answer the question

You can use the following tools:

1. entity_matcher(node_name="query", graph_type="technology")
   - Purpose: Finds entities in the knowledge graph that match or are similar to your query
   - Parameters:
     * node_name: The entity or concept you want to search for (e.g., "crop diseases", "legal precedent")
     * graph_type: Must be "technology" for this question
   - Returns: A list of matching entity names in the knowledge graph

2. node_info(node_name="precise entity name", graph_type="technology")
   - Purpose: Retrieves detailed information about a specific entity and its relationships
   - Parameters:
     * node_name: The exact entity name (use names returned by entity_matcher)
     * graph_type: Must be "technology" for this question
   - Returns: Entity properties and a list of relationships this entity has with others

Example of the iterative process:

<think>
I need to find information about agriculture practices. Let me first search for entities related to agriculture.
</think>
<search>entity_matcher(node_name="Agriculture", graph_type="agriculture")</search>
<information>HERE IS THE TOOL RETURNED RESULTS</information>
<think>
Now I see several agricultural entities. I should get more details about "Sustainable Agriculture" to understand its relationships.
</think>
<search>node_info(node_name="Sustainable Agriculture", graph_type="agriculture")</search>
<information>HERE IS THE TOOL RETURNED RESULTS</information>
<think>
Based on the information I've gathered, I now have enough details to answer the question.
</think>
<answer>
HERE IS YOUR ANSWER
</answer>

Your answer should only contain one entity name. Don't include any other text.

Question: {question}<|im_end|>
<|im_start|>assistant
"""

        dialog = system_prompt + "\n" + user_prompt
        dialogs.append(dialog)
        questions.append(question)
        golden_answers.append(golden_answer)
        graph_types.append(graph_type)

    max_rounds = 7
    finished = [False] * len(dialogs)

    for round_idx in range(max_rounds):
        inputs = [dialogs[i] for i in range(len(dialogs)) if not finished[i]]
        active_indices = [i for i in range(len(dialogs)) if not finished[i]]
        
        print("INFO: inputs: ============== ")
        print(inputs)
        
        if not inputs:
            break
        try:
            outputs = model.generate(inputs, sampling_params=sampling_params)
        except Exception as e:
            print(f"生成错误: {str(e)}")
            continue
        

        # 收集所有工具调用请求
        tool_requests = []
        tool_matches = []

        for idx, output in enumerate(outputs):
            i = active_indices[idx]

            # 如果之前已经完成，就跳过所有逻辑（防御性判断）
            if finished[i]:
                continue

            # generated_text = output.outputs[0].text if output.outputs else ""
            raw_ids = output.outputs[0].token_ids
            generated_text = tokenizer.decode(raw_ids) if raw_ids else ""
            dialogs[i] += f"{generated_text}"
            
            print("INFO: outputs: ============== ")
            print(generated_text)

            # 检查是否已生成最终答案
            if "<answer>" in generated_text and "</answer>" in generated_text:
                finished[i] = True
                print("Info: finished answer:")
                print(generated_text)
                continue  # 标记为完成，后续跳过处理

            # 根据 use_json 分支选择不同的匹配方式
            if use_json:
                match = re.search(r"<search>\s*(\{.*?\})\s*</search>", generated_text, re.DOTALL)
            else:
                match = re.search(r"<search>\s*([a-zA-Z0-9_]+\s*\(.*?\))\s*</search>", generated_text, re.DOTALL)

            if not finished[i]:  # 再次确保只对未完成样本处理
                tool_matches.append((i, match, generated_text))

                if match:
                    if use_json:
                        # 使用json解析
                        search_str = match.group(1)
                        try:
                            search_json = json.loads(search_str)
                            tool_name = search_json["name"]
                            node_name = search_json["parameters"]["node_name"]
                            graph_type_extracted = search_json["parameters"]["graph_type"]
                            tool_requests.append((i, tool_name, node_name, graph_type_extracted))
                        except json.JSONDecodeError as e:
                            print(f"JSON解析失败: {search_str}")
                            print(f"错误信息: {str(e)}")
                            tool_requests.append((i, "", "", ""))
                        except (KeyError, ValueError) as e:
                            print(f"工具调用格式错误: {search_str}")
                            print(f"错误信息: {str(e)}")
                            tool_requests.append((i, "", "", ""))
                    else:
                        # 使用正则表达式解析
                        try:
                            tool_name = re.search(r"<search>\s*([a-zA-Z0-9_]+)\s*\(", generated_text).group(1)
                            node_name = re.search(r'node_name="(.*?)"', generated_text).group(1)
                            graph_type_extracted = re.search(r'graph_type="(.*?)"', generated_text).group(1)
                            tool_requests.append((i, tool_name, node_name, graph_type_extracted))
                        except (AttributeError, ValueError) as e:
                            print(f"正则表达式解析失败: {generated_text}")
                            print(f"错误信息: {str(e)}")
                            tool_requests.append((i, "", "", ""))

        # 仅对未完成推理的样本进行工具调用
        filtered_tool_requests = [(i, tool_name, node_name, graph_type) for (i, tool_name, node_name, graph_type) in tool_requests if not finished[i]]
        # 并发执行工具调用
        informations = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(call_tool, tool_name, node_name, graph_type): i for i, tool_name, node_name, graph_type in filtered_tool_requests}
            for future in futures:
                i = futures[future]
                informations[i] = future.result()

        # 追加工具响应到对话中
        for i, match, generated_text in tool_matches:
            if match and i in informations and not finished[i]:
                dialogs[i] += f"<information>{informations[i]}</information>\n\n"

        if all(finished):
            break
            

    # 统计准确率
    for i, dialog in enumerate(dialogs):
        final_answer = ""
        try:
            # 使用与训练时相同的答案提取方法
            final_answer = extract_solution(dialog)
            if final_answer is None:
                final_answer = ""
        except:
            final_answer = ""

        result = {
            "question": questions[i],
            "golden_answer": golden_answers[i],
            "predicted_answer": final_answer,
            "graph_type": graph_types[i],
            "dialog": dialog,
        }
        batch_results.append(result)

        stats["processed"] += 1
        
        # 使用与训练时相同的评估方法
        is_correct = False
        if final_answer:
            normalized_pred = normalize_answer(final_answer)
            # golden_answers[i] 可能是字符串或列表
            golden_targets = golden_answers[i] if isinstance(golden_answers[i], list) else [golden_answers[i]]
            for target in golden_targets:
                if normalized_pred == normalize_answer(target):
                    is_correct = True
                    break
        
        if is_correct:
            stats["correct"] += 1
        accuracy = stats["correct"] / stats["processed"] if stats["processed"] > 0 else 0
        print(f"[实时统计] 已处理: {stats['processed']}，正确数: {stats['correct']}，当前准确率: {accuracy:.4f}")

    save_buffer_to_file(output_path, batch_results)

# ============= 主函数 =============
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行知识图谱推理')
    parser.add_argument('--type', type=str, choices=['12hop', '34hop'], required=True,
                      help='选择测试集类型：12hop 或 34hop')
    parser.add_argument('--think', action='store_true', help='使用思考过程')
    parser.add_argument('--no-think', dest='think', action='store_false', help='不使用思考过程')

    parser.add_argument('--json', action='store_true', help='使用 JSON 工具调用格式')
    parser.add_argument('--no-json', dest='json', action='store_false', help='使用函数调用格式')

    parser.set_defaults(think=True, json=True)
    args = parser.parse_args()

    print(f"知识图谱工具初始化完成，endpoint: {endpoint}")

    dataset, _ = load_dataset(args)
    model, lora_request, tokenizer = init_model()

    sampling_params = SamplingParams(
        temperature=0,
        top_k=1,
        # temperature=0.5,
        # top_p=0.95,
        max_tokens=3000,
        skip_special_tokens=True,
        stop = ["</search>"]
    )

    batch_size = 1  # TODO:根据 GPU 情况可调
    stats = {"correct": 0, "processed": 0}
    with tqdm(total=len(dataset), desc="处理样本（条）") as pbar:
        for i in range(0, 10, batch_size): # len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            batch_process_samples(batch, model, sampling_params, lora_request, stats, tokenizer, args.think, args.json)
            pbar.update(len(batch))

    print(f"全部推理完成！结果保存在 {output_path}")

if __name__ == "__main__":
    """
    # 推理12hop数据集，使用思考过程，使用json解析工具调用
    # python run_scripts/vllm_inference_kg_batch.py --type 12hop
    # 不使用json格式：python run_scripts/vllm_inference_kg_batch.py --type 12hop --no-json
    """
    main()