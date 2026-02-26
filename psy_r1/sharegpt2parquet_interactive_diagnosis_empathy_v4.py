#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式问诊数据转换脚本 v4

将 SMHC MiroDiag 数据转换为交互式问诊训练格式

v4 相比 v2 的关键修改：
- 新增 patient_info 字段：包含患者的完整病例信息（主诉、现病史、个人史等）
- patient_info 用于 SIG (Shapley Information Gain) 过程奖励计算
- 同时在 ask_patient 和 do_diagnose 的 create_kwargs 中添加 patient_info
- 输出目录改为 v4 版本

使用方法：
    python sharegpt2parquet_interactive_diagnosis_empathy_v4.py

    # 或指定参数
    python sharegpt2parquet_interactive_diagnosis_empathy_v4.py \
        --input /path/to/input.json \
        --output /path/to/output_dir \
        --train-ratio 0.9
"""

import json
import pandas as pd
import re
import os
import argparse
from typing import Dict, List, Any, Optional


def extract_major_icd_codes(full_icd_code: str) -> List[str]:
    """
    从完整 ICD 代码中提取大类代码（前3位）

    Args:
        full_icd_code: 完整的 ICD 代码，例如 'F20.400'

    Returns:
        大类代码列表，例如 ['F20']
    """
    if not full_icd_code:
        return []

    # 提取 F 开头的三位大类代码
    match = re.match(r'(F\d{2})', full_icd_code)
    if match:
        return [match.group(1)]
    return []


def build_patient_info(data_item: Dict[str, Any]) -> str:
    """
    从原始数据构建完整的 patient_info 文本

    v4 新增：用于 SIG 过程奖励计算

    Args:
        data_item: 原始数据项

    Returns:
        患者信息文本，包含主诉、现病史、个人史等
    """
    # 优先使用原始数据中已有的 Patient info 字段
    if 'Patient info' in data_item and data_item['Patient info']:
        return data_item['Patient info']

    # 如果没有 Patient info 字段，则手动构建
    info_parts = []

    # 基本信息
    if data_item.get('Age'):
        info_parts.append(f"年龄: {data_item['Age']}岁")
    if data_item.get('Gender'):
        info_parts.append(f"性别: {data_item['Gender']}")

    # 主诉
    if data_item.get('ChiefComplaint'):
        info_parts.append(data_item['ChiefComplaint'])

    # 现病史
    if data_item.get('PresentIllnessHistory'):
        info_parts.append(data_item['PresentIllnessHistory'])

    # 个人史
    if data_item.get('PersonalHistory'):
        info_parts.append(data_item['PersonalHistory'])

    # 家族史
    if data_item.get('FamilyHistory'):
        info_parts.append(f"家族史: {data_item['FamilyHistory']}")

    # 躯体疾病史
    if data_item.get('ImportantRelevantPhysicalIllnessHistory'):
        info_parts.append(data_item['ImportantRelevantPhysicalIllnessHistory'])

    # 药物过敏史
    if data_item.get('DrugAllergyHistory'):
        info_parts.append(f"药物过敏史: {data_item['DrugAllergyHistory']}")

    return '\n'.join(info_parts)


def create_interactive_diagnosis_prompt_v4() -> List[Dict[str, str]]:
    """
    创建交互式问诊的 system prompt + 初始 user message (v4)

    与 v2 保持一致，仅作版本标记

    Returns:
        包含 system prompt 和初始 user message 的消息列表
    """
    system_prompt = """你是一名精神卫生中心的临床心理科医师，进行专业问诊。
遵循 ICD-10 精神与行为障碍标准，先通过充分的共情式问诊收集信息，再进行诊断。

## 核心问诊原则

1. **建立治疗性关系**：
   - 在提问时表达对患者困扰的理解和共情
   - 使用支持性语言，如"我能理解这一定让您很困扰"、"听起来您经历了很艰难的时期"
   - 避免冷漠的连续提问，在关键信息后给予情感回应

2. **深入追问症状细节**：
   - 对患者描述的每个症状进行具体化：持续时间、频率、严重程度、触发因素
   - 询问症状的时间进程：何时开始、是否加重、是否有缓解期
   - 探查症状对日常功能的影响：工作、人际关系、自我照顾能力
   - 使用开放式问题引导患者详细描述，如"能详细说说吗"、"还有其他表现吗"

3. **验证症状真实性**：
   - 对患者自述的症状进行交叉验证：询问具体事例、客观表现
   - 观察症状描述的一致性和逻辑性
   - 区分患者的主观感受与客观症状

4. **鉴别诊断**：
   - 系统性排查相似疾病的鉴别点
   - 针对性询问关键区分症状（如抑郁与双相的既往躁狂史、焦虑与惊恐的发作模式）
   - 探查器质性因素：用药史、躯体疾病、物质使用
   - 评估共病可能：多个诊断可能并存时需要分别评估

## ICD-10 精神与行为障碍标准大类

请仅从以下ICD-10标准中的10种疾病中选择最符合的诊断大类以及进一步细分的ICD-10小类：

- F32 抑郁发作：情绪持续低落、兴趣/愉快感下降、精力不足；伴睡眠/食欲改变、自责/无价值感等；可轻/中/重度（重度可伴精神病性症状）；无既往躁狂/轻躁狂。
  *鉴别要点：需排除双相障碍（F31）、适应障碍（F43）、器质性情绪障碍*

- F41 其他焦虑障碍：恐慌发作或广泛性焦虑为主；过度担忧、紧张、心悸、胸闷、出汗、眩晕、濒死感/失控感；与特定情境无关或不成比例，造成显著痛苦/功能损害。
  *鉴别要点：需排除躯体疾病（如甲亢）、物质诱发、抑郁伴焦虑*

- F39.9 未特指的心境（情感）障碍：存在心境障碍证据，但资料不足以明确归入抑郁或双相等具体亚型时选用。

- F51 非器质性睡眠障碍：失眠、过度嗜睡、梦魇、昼夜节律紊乱等；非器质性原因；睡眠问题为主要主诉并致显著困扰/功能损害。
  *鉴别要点：需排除抑郁/焦虑继发失眠、睡眠呼吸暂停等器质性原因*

- F98 其他儿童和青少年行为与情绪障碍：多见于儿童期起病（如遗尿/遗粪、口吃、抽动相关习惯性问题等），以发育期特异表现为主。

- F42 强迫障碍：反复的强迫观念/行为，个体自知过度或不合理但难以抵抗，耗时或致显著困扰/损害。
  *鉴别要点：需排除精神分裂症的思维障碍、抑郁的反刍思维*

- F31 双相情感障碍：既往或目前存在躁狂/轻躁狂发作与抑郁发作的交替或混合；需有明确躁狂谱系证据。
  *鉴别要点：详细询问既往是否有情绪高涨、精力旺盛、睡眠需求减少、冲动行为等躁狂症状*

- F43 对严重应激反应和适应障碍：与明确应激事件有关；可为急性应激反应、PTSD或适应障碍；核心包含再体验、回避、警觉性增高或与应激源相关的情绪/行为改变。
  *鉴别要点：需确认应激源存在、症状与应激源的时间关联、是否超出正常应激反应*

- F45 躯体形式障碍：反复或多样躯体症状为主（如疼痛、心悸、胃肠不适等），检查难以找到足以解释的器质性原因或与病因不相称，显著痛苦/就诊反复。
  *鉴别要点：需详细了解躯体检查结果、症状与情绪的关联、疾病信念*

- F20 精神分裂症：在知觉、思维、情感及行为等方面的广泛障碍；常见持续性妄想、幻听、思维松弛/破裂、情感淡漠、阴性症状，病程≥1月（或依本地标准）。
  *鉴别要点：需评估现实检验能力、思维连贯性、情感协调性*

- Z71 咨询和医疗建议相关因素：包括心理咨询、健康教育、生活方式指导等，当患者主要需要咨询服务而非特定疾病治疗时使用。

## 问诊流程建议

1. **开场阶段**（1-2轮）：表达欢迎，了解主诉，建立初步信任
2. **症状探查阶段**（8-15轮）：深入了解核心症状及伴随症状，充分细节追问
3. **鉴别诊断阶段**（3-5轮）：针对性询问鉴别点，排除其他可能诊断
4. **功能评估阶段**（2-3轮）：评估症状对生活功能的影响
5. **诊断决策阶段**：综合信息，给出诊断

## 工具调用规范

你必须通过调用工具来完成问诊和诊断，每一轮输出的格式如下：

1) **问诊阶段**（信息不足时）：先在 <think> 标签中写出思考过程，然后调用 ask_patient 工具向患者提问。
   示例：
   <think>患者提到情绪低落，需要了解持续时间和严重程度</think>
   <tool_call>
   {"name": "ask_patient", "arguments": {"question": "我能理解您现在很不舒服。这种情绪低落的感觉大概持续多久了？"}}
   </tool_call>

2) **诊断阶段**（信息充分时）：先在 <think> 标签中写出完整的诊断推理，然后调用 do_diagnose 工具提交诊断。
   diagnosis 参数中只放 <box>ICD-10代码</box>，诊断推理写在 <think> 中。
   示例：
   <think>患者表现为持续情绪低落、兴趣减退、睡眠障碍，符合抑郁发作的诊断标准。排除双相障碍和适应障碍。</think>
   <tool_call>
   {"name": "do_diagnose", "arguments": {"diagnosis": "<box>F32.1</box>"}}
   </tool_call>

## 重要提示

- 当信息不足以排除其他疾病进行诊断时，必须继续调用 ask_patient 进行深入问诊
- 避免过早下诊断，确保收集到充分的鉴别信息
- 当确信可以诊断时，调用 do_diagnose，diagnosis 参数中必须包含 <box>ICD-10代码</box>
- ICD-10小类可多项，用分号分隔，如 <box>F32.1; F41.2</box>
- 保持温暖、专业的医患沟通风格，在专业性与人文关怀之间取得平衡"""

    return [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": "你好医生。"
        }
    ]


def make_interactive_map_fn_v4(data_source: str, split: str, patient_version: str = "v3", model_name: str = "Qwen3-32B"):
    """
    创建交互式问诊数据转换函数 (v4 - 带 patient_info)

    Args:
        data_source: 数据源标识
        split: 'train' 或 'val'
        patient_version: Patient Agent 版本
        model_name: Patient Agent 使用的模型

    Returns:
        数据转换函数
    """
    def process_fn(data_item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """处理单个数据项"""
        # 提取 patient_id 和诊断代码
        patient_id = data_item.get('patient_id')
        full_icd_code = data_item.get('DiagnosisCode', '')

        # 过滤无效数据
        if not patient_id or not full_icd_code:
            return None

        # 提取大类 ICD 代码
        major_icd_codes = extract_major_icd_codes(full_icd_code)

        if not major_icd_codes:
            return None

        # v4 新增：构建 patient_info
        patient_info = build_patient_info(data_item)

        # 创建 prompt
        prompt = create_interactive_diagnosis_prompt_v4()

        # 构建转换后的数据
        converted_data = {
            "data_source": data_source,
            "prompt": prompt,
            "raw_prompt": prompt,
            "ability": "interactive_diagnosis",
            "reward_model": {
                "style": "rule",
                "ground_truth": major_icd_codes
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "visit_number": patient_id,
                "full_icd_code": [full_icd_code],
                "diagnosis_name": data_item.get('Diagnosis', ''),
                "patient_id": patient_id,
                # v4 新增：顶层 patient_info
                "patient_info": patient_info,
                "tools_kwargs": {
                    "ask_patient": {
                        "create_kwargs": {
                            "patient_id": patient_id,
                            "patient_version": patient_version,
                            "model_name": model_name,
                            # v4 新增：在 ask_patient 中也传递 patient_info
                            "patient_info": patient_info,
                        }
                    },
                    "do_diagnose": {
                        "create_kwargs": {
                            "patient_id": patient_id,
                            "ground_truth": major_icd_codes,
                            "patient_version": patient_version,
                            "model_name": model_name,
                            # v4 新增：在 do_diagnose 中传递 patient_info（用于 SIG 计算）
                            "patient_info": patient_info,
                        }
                    }
                }
            }
        }

        return converted_data

    return process_fn


def convert_to_interactive_diagnosis_parquet_v4(
    train_file: str,
    output_dir: str,
    train_ratio: float = 0.9,
    max_samples: Optional[int] = None,
    patient_version: str = "v3",
    model_name: str = "Qwen3-32B",
):
    """
    将数据转换为交互式问诊训练格式的 parquet 文件 (v4)

    Args:
        train_file: 输入 JSON 文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例
        max_samples: 最大样本数（用于调试）
        patient_version: Patient Agent 版本
        model_name: Patient Agent 使用的模型

    Returns:
        (train_df, val_df): 训练集和验证集 DataFrame
    """
    data_source = "SMHC_RL_interactive_diagnosis_v4"

    print(f"正在读取数据: {train_file}")
    with open(train_file, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    print(f"  读取到 {len(all_data)} 条数据")

    if max_samples:
        all_data = all_data[:max_samples]
        print(f"  限制样本数为 {max_samples}")

    # 去重：基于 patient_id
    print("正在按 patient_id 去重...")
    seen_ids = set()
    deduplicated_data = []

    for item in all_data:
        patient_id = item.get('patient_id', '')
        if patient_id and patient_id not in seen_ids:
            seen_ids.add(patient_id)
            deduplicated_data.append(item)

    print(f"去重后: {len(all_data)} -> {len(deduplicated_data)} 条")

    # 按比例划分训练集和验证集
    split_idx = int(len(deduplicated_data) * train_ratio)
    train_data = deduplicated_data[:split_idx]
    val_data = deduplicated_data[split_idx:]

    print(f"数据划分: 训练集 {len(train_data)} 条, 验证集 {len(val_data)} 条")

    # 创建转换函数
    train_map_fn = make_interactive_map_fn_v4(data_source, 'train', patient_version, model_name)
    val_map_fn = make_interactive_map_fn_v4(data_source, 'val', patient_version, model_name)

    # 转换训练数据
    print("正在转换训练数据...")
    converted_train = []
    filtered_train_count = 0
    patient_info_lengths = []

    for idx, item in enumerate(train_data):
        result = train_map_fn(item, idx)
        if result is not None:
            converted_train.append(result)
            # 统计 patient_info 长度
            pi = result['extra_info'].get('patient_info', '')
            patient_info_lengths.append(len(pi))
        else:
            filtered_train_count += 1

    # 转换验证数据
    print("正在转换验证数据...")
    converted_val = []
    filtered_val_count = 0

    for idx, item in enumerate(val_data):
        result = val_map_fn(item, idx)
        if result is not None:
            converted_val.append(result)
        else:
            filtered_val_count += 1

    print(f"过滤统计:")
    print(f"  训练集: 原始 {len(train_data)} 条, 过滤 {filtered_train_count} 条, 保留 {len(converted_train)} 条")
    print(f"  验证集: 原始 {len(val_data)} 条, 过滤 {filtered_val_count} 条, 保留 {len(converted_val)} 条")

    # v4 新增：统计 patient_info 信息
    if patient_info_lengths:
        avg_len = sum(patient_info_lengths) / len(patient_info_lengths)
        max_len = max(patient_info_lengths)
        min_len = min(patient_info_lengths)
        empty_count = sum(1 for l in patient_info_lengths if l == 0)
        print(f"\npatient_info 统计:")
        print(f"  平均长度: {avg_len:.1f} 字符")
        print(f"  最大长度: {max_len} 字符")
        print(f"  最小长度: {min_len} 字符")
        print(f"  空值数量: {empty_count} ({empty_count/len(patient_info_lengths)*100:.2f}%)")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 转换为 DataFrame 并保存为 parquet
    print("\n正在保存为 parquet 格式...")
    train_df = pd.DataFrame(converted_train)
    val_df = pd.DataFrame(converted_val)

    train_output_path = os.path.join(output_dir, 'train.parquet')
    val_output_path = os.path.join(output_dir, 'val.parquet')

    train_df.to_parquet(train_output_path, index=False)
    val_df.to_parquet(val_output_path, index=False)

    print(f"训练数据已保存至: {train_output_path}")
    print(f"验证数据已保存至: {val_output_path}")
    print(f"训练数据量: {len(converted_train)} 条, 验证数据量: {len(converted_val)} 条")

    # 打印样例
    if len(converted_train) > 0:
        print("\n" + "=" * 80)
        print("样例数据 (训练集第一条):")
        print("=" * 80)
        sample = converted_train[0]
        # 简化输出，不打印完整 prompt
        print(f"data_source: {sample['data_source']}")
        print(f"ability: {sample['ability']}")
        print(f"reward_model: {sample['reward_model']}")
        print(f"\nextra_info.patient_id: {sample['extra_info']['patient_id']}")
        print(f"extra_info.diagnosis_name: {sample['extra_info']['diagnosis_name']}")
        print(f"extra_info.full_icd_code: {sample['extra_info']['full_icd_code']}")
        print(f"\n[v4 新增] extra_info.patient_info (前500字符):")
        pi = sample['extra_info'].get('patient_info', '')
        print(f"  {pi[:500]}..." if len(pi) > 500 else f"  {pi}")
        print(f"\n[v4 新增] tools_kwargs.do_diagnose.create_kwargs 包含 patient_info: {'patient_info' in sample['extra_info']['tools_kwargs']['do_diagnose']['create_kwargs']}")

    return train_df, val_df


def main():
    parser = argparse.ArgumentParser(description='交互式问诊数据转换脚本 v4')
    parser.add_argument('--input', '-i', type=str,
                        default="/tcci_mnt/shihao/project/Lingxi_annotation_0111/raw_data/LingxiDiag-16K_train_data.json",
                        help='输入 JSON 文件路径')
    parser.add_argument('--output', '-o', type=str,
                        default="/tcci_mnt/shihao/project/verl/psy_r1/dataset_rl/LingxiDiag-interactive-long-empathy-v4",
                        help='输出目录')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                        help='训练集比例 (默认 0.9)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='最大样本数（用于调试）')
    parser.add_argument('--patient-version', type=str, default="v3",
                        help='Patient Agent 版本 (默认 v3)')
    parser.add_argument('--model-name', type=str, default="Qwen3-32B",
                        help='Patient Agent 使用的模型 (默认 Qwen3-32B)')

    args = parser.parse_args()

    print("=" * 80)
    print("交互式问诊数据转换脚本 v4")
    print("=" * 80)
    print(f"v4 新特性:")
    print(f"  - 新增 patient_info 字段（用于 SIG 过程奖励计算）")
    print(f"  - patient_info 包含：主诉、现病史、个人史、家族史等")
    print(f"  - 同时在 ask_patient 和 do_diagnose 的 create_kwargs 中添加 patient_info")
    print("=" * 80)
    print(f"输入文件: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"训练集比例: {args.train_ratio}")
    print(f"Patient Agent 版本: {args.patient_version}")
    print(f"Patient Agent 模型: {args.model_name}")
    if args.max_samples:
        print(f"限制样本数: {args.max_samples}")
    print("=" * 80)

    train_df, val_df = convert_to_interactive_diagnosis_parquet_v4(
        train_file=args.input,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        max_samples=args.max_samples,
        patient_version=args.patient_version,
        model_name=args.model_name,
    )

    print("\n" + "=" * 80)
    print("转换完成！")
    print("=" * 80)

    # 数据验证
    print("\n数据验证:")
    print(f"训练集数据量: {len(train_df)}")
    print(f"验证集数据量: {len(val_df)}")
    print(f"数据列名: {train_df.columns.tolist()}")

    # 统计诊断代码分布
    print("\n诊断代码分布 (训练集 前10):")
    train_diagnosis = train_df['reward_model'].apply(lambda x: str(x['ground_truth']))
    print(train_diagnosis.value_counts().head(10))

    # 验证 patient_info 是否正确添加
    sample = train_df.iloc[0]['extra_info']
    print(f"\n[验证] extra_info 包含 patient_info: {'patient_info' in sample}")
    print(f"[验证] tools_kwargs.ask_patient.create_kwargs 包含 patient_info: {'patient_info' in sample['tools_kwargs']['ask_patient']['create_kwargs']}")
    print(f"[验证] tools_kwargs.do_diagnose.create_kwargs 包含 patient_info: {'patient_info' in sample['tools_kwargs']['do_diagnose']['create_kwargs']}")

    print("\n" + "=" * 80)
    print("数据已准备好用于交互式问诊训练 (v4 - 支持 SIG 过程奖励)")
    print("=" * 80)


if __name__ == "__main__":
    main()
