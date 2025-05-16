"""
测试extract_thinking_content函数在真实LLM输出上的效果
"""
import os
import sys
from pathlib import Path
import time
import json

# 添加项目根目录到系统路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))


os.environ["USE_LOCAL_QWEN_FOR_EVAL"] = "true"
os.environ["LOCAL_QWEN_MODEL"] = "/mnt/tanka/models/Qwen2.5-32B-Instruct"
os.environ["VLLM_API_BASE"] = "http://10.119.21.75:9001/v1"


from verl.utils.kg_rewards import extract_thinking_content, get_model_name, extract_tag_content

def call_vllm_qwen(prompt: str, model_name: str = None, max_tokens: int = 1024) -> str:
    """
    调用本地已部署的vllm Qwen模型服务（普通文本响应方式）
    """
    # 如果未指定模型名称，使用配置的默认值
    model_name = os.getenv("LOCAL_QWEN_MODEL", "/mnt/tanka/models/Qwen2.5-32B-Instruct")
        
    try:
        from openai import OpenAI
        
        # 从环境变量获取API基础URL，默认为本地vLLM服务
        api_base = os.getenv("VLLM_API_BASE", "http://10.119.21.75:9001/v1")
        
        print(f"正在使用模型: {model_name}")
        print(f"API基础URL: {api_base}")
        
        # 创建OpenAI客户端连接到vLLM服务
        client = OpenAI(
            api_key="EMPTY",  # vLLM服务通常不需要API密钥
            base_url=api_base,
        )
        
        # 调用聊天接口
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            top_p=0.9,
            max_tokens=max_tokens,
            timeout=30,
        )
        
        # 提取生成的文本
        generated_text = chat_response.choices[0].message.content
        
        return generated_text
    except Exception as e:
        print(f"vLLM调用失败: {str(e)}")
        return ""

def test_think_tag_generation():
    """测试LLM生成带有think标签的内容并提取"""
    prompts = [
        # 知识图谱提取任务的提示
        """Chunk Text:
[msg_id: 0] 2025-03-18T16:43:47 Huazheng WU: 大家好，今天服务端已升级Google Gemini2.0的语音消息转文字功能（感谢@Yize CHEN @Yuchen ZANG 推荐Gemini作为调研方向），欢迎体验并反馈使用感受！

在调研对比传统供应商（Google Speech、科大讯飞、Microsoft、Fano）与AI大模型（Gladia、AssemblyAI、Deepgram、Gemini）后，我们从准确性、响应速度、成本、性能、自动语言识别、音频格式支持等多个维度进行了综合评估，Gemini在ASR（自动语音识别）方面表现突出。

此外，我们还支持了语音情绪检测，并通过emoji直观呈现情绪状态。Google Speech语音转文字将作为备用通道，确保系统稳定性。
[msg_id: 1] 2025-03-18T22:14:33 Tianqiao Chen: 我来试试
[msg_id: 2] 2025-03-18T22:58:28 Tianqiao Chen: 谁给我发一段长一点的语音：）
[msg_id: 3] 2025-03-18T23:00:47 Huazheng WU: sent an audio
[msg_id: 4] 2025-03-18T23:04:14 Hua ZHANG: sent an audio
[msg_id: 5] 2025-03-18T23:05:35 Hua ZHANG: 说的比较零碎含糊，转出来还是比较还原的
[msg_id: 6] 2025-03-18T23:05:37 Tianqiao Chen: 又快又好
[msg_id: 7] 2025-03-18T23:06:12 Tianqiao Chen: 可见AI时代放弃传统思路是非常必要的
[msg_id: 8] 2025-03-18T23:06:30 Tianqiao Chen: 看看还有什么可以革新的
[msg_id: 9] 2025-03-18T23:06:48 Tianqiao Chen: 搜索就是下一个了
[msg_id: 10] 2025-03-18T23:06:58 Hua ZHANG: 是的，耐心测试对比，还是能实现我们要的效果😄
[msg_id: 11] 2025-03-18T23:25:15 Tianqiao Chen: 如果能把嗯，啊这些口语化去掉就更好了，就像我说smart reply用语音修改一样
[msg_id: 12] 2025-03-18T23:25:26 Tianqiao Chen: 不过可能不够准确😀
[msg_id: 13] 2025-03-18T23:28:32 Hua ZHANG: 当前特意转的和原始语音内容一摸一样。我们可以调整prompt试试修饰效果。但是修饰了会不会就改变了功能特点了，因为我们也可以加上翻译，让AI直接转成其他语种
[msg_id: 14] 2025-03-18T23:29:08 Tianqiao Chen: 是的
[msg_id: 15] 2025-03-18T23:29:32 Tianqiao Chen: 语音这边可以做的功能挺多的，你们技术部门想想怎么样做成一个亮点功能
[msg_id: 16] 2025-03-18T23:30:15 Hua ZHANG: 好的，我们多试试不同效果看看

Instructions: Analyze the above text, explain your reasoning inside <think> tags, and output a structured knowledge graph in JSON format inside <answer> tags. The JSON should have two keys: 'nodes' and 'edges'. For example:
{"nodes": [{"id": 1, "label": "EntityName", "type": "EntityType"}, ...],"edges": [{"source": 1, "target": 2, "relation": "RELATION_TYPE", "description": "Explanation of the relationship", "keywords": ["key1", "key2","key3"], "strength": 8, "msg_ids": [0, 1, 2]}, ...]}
Requirements:
- Ensure your output strictly follows this format.
- For each relationship/edge, include:
    - source: ID of the source entity
    - target: ID of the target entity
    - relation: Only one type of relationship for each edge
    - description: Detailed explanation of why the entities are related
    - keywords: List of key terms that can be searched or represent important elements of the relationship
    - strength: Numeric score (1-10) indicating relationship strength
    - msg_ids: List of message IDs where this relationship was mentioned
- The output language is English. For specific entities (person name, product name), use the the original language of in the text.
- msg_ids can not be empty.""",
    ]
    
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n\n测试 {i+1}: {'引导使用think标签' if '<think>' in prompt else '不引导使用think标签'}")
        print("-" * 50)
        
        # 生成回答
        print("正在请求LLM生成回答...")
        start_time = time.time()
        response = call_vllm_qwen(prompt)
        duration = time.time() - start_time
        print(f"生成完成，耗时: {duration:.2f}秒")
        
        # 尝试提取思考内容
        thinking = extract_thinking_content(response)
        
        # 检查是否使用了think标签
        has_think_tag = "<think>" in response.lower() and "</think>" in response.lower()
        
        result = {
            "prompt_type": "引导使用think标签" if "<think>" in prompt else "不引导使用think标签",
            "response_length": len(response),
            "has_think_tag": has_think_tag,
            "thinking_length": len(thinking),
            "thinking_extracted": bool(thinking),
        }
        
        results.append(result)
        
        # 打印结果摘要
        print("\n结果摘要:")
        print(f"回答长度: {len(response)} 字符")
        print(f"是否包含think标签: {'是' if has_think_tag else '否'}")
        print(f"提取到的思考内容长度: {len(thinking)} 字符")
        print(f"是否成功提取思考内容: {'是' if thinking else '否'}")
        
        # 打印回答的前100个字符
        print("\n回答摘要 (前100个字符):")
        print(response[:100] + "..." if len(response) > 100 else response)
        
        # 如果提取到思考内容，打印前100个字符
        if thinking:
            print("\n提取到的思考内容 (前100个字符):")
            print(thinking[:100] + "..." if len(thinking) > 100 else thinking)
        
        print("-" * 50)
    
    # 打印总结
    print("\n\n测试总结:")
    print("-" * 50)
    for i, result in enumerate(results):
        print(f"测试 {i+1} ({result['prompt_type']}):")
        print(f"  回答长度: {result['response_length']} 字符")
        print(f"  是否包含think标签: {'是' if result['has_think_tag'] else '否'}")
        print(f"  提取到的思考内容长度: {result['thinking_length']} 字符")
        print(f"  是否成功提取思考内容: {'是' if result['thinking_extracted'] else '否'}")
    
    # 保存结果到JSON文件
    try:
        with open("llm_thinking_extraction_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("\n结果已保存到 llm_thinking_extraction_results.json")
    except Exception as e:
        print(f"\n保存结果失败: {str(e)}")

if __name__ == "__main__":
    test_think_tag_generation() 