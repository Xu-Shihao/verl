# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
知识图谱抽取的PPO训练脚本，基于verl框架实现
"""

import os
import argparse
import sys

import hydra
import ray

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
# from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.kg_rewards import compute_kg_extraction_reward
from verl.utils.dataset.rl_dataset import collate_fn

import dotenv
dotenv.load_dotenv()

def get_custom_reward_fn(config):
    """
    从指定路径动态加载自定义奖励函数。
    
    参数:
        config: 包含自定义奖励函数配置的字典
        
    返回:
        wrapped_fn: 带有额外参数的包装奖励函数
    """
    import importlib.util
    import sys

    # 获取自定义奖励函数配置
    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"奖励函数文件 '{file_path}' 未找到.")

    # 动态导入模块
    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"从 '{file_path}' 加载模块时出错: {e}") from e

    # 获取指定的函数
    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"奖励函数 '{function_name}' 在文件 '{file_path}' 中未找到.")

    print(f"使用自定义奖励函数 '{function_name}' (来自 '{file_path}')")
    raw_fn = getattr(module, function_name)

    # 获取奖励函数的额外参数
    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))
    
    # 创建包装函数，注入额外参数
    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


@hydra.main(config_path="config", config_name="kg_extraction_ppo_trainer", version_base=None)
def main(config):
    """主入口函数，使用hydra加载配置并启动知识图谱抽取的PPO训练"""
    try:
        # 解析命令行参数，但不使用sys.argv（因为hydra已经处理过这些参数）
        # 仅用于打印调试信息和获取默认值
        args = argparse.ArgumentParser(description="知识图谱抽取的PPO训练")
        args.add_argument("--config_path", type=str, default="/mnt/afs/tanka/shihao/project/verl/verl/trainer/config/kg_extraction_ppo_trainer_debug.yaml",
                         help="自定义配置文件路径，覆盖默认配置")
        args.add_argument("--model_path", type=str, default="/mnt/afs/tanka/shihao/model/Qwen2.5-7B-Instruct",
                         help="基础模型路径，覆盖配置文件中的设置")
        args.add_argument("--output_dir", type=str, default="/mnt/afs/tanka/shihao/outputs/kg_extraction",
                         help="输出目录，覆盖配置文件中的设置")
        args.add_argument("--debug", action="store_true",
                         help="调试模式，使用较小的训练配置")
        args.add_argument("--quick_test", action="store_true",
                         help="快速测试模式，仅初始化训练环境但不开始训练")
        args.add_argument("--debug_locally", action="store_true",
                         help="本地调试模式，在主进程中运行TaskRunner，便于断点调试")
        args = args.parse_args([])  # 仅获取默认值，不处理实际命令行参数
        
        # 打印调试信息
        print_debug_info(config, args)
    except Exception as e:
        print(f"调试信息处理失败: {e}")
    
    # 应用快速测试模式标志（如果需要）
    if "quick_test_mode" in config and config.quick_test_mode:
        print("启用快速测试模式...")
    
    run_ppo(config)


def run_ppo(config) -> None:
    """
    运行PPO训练的主函数
    
    参数:
        config: 包含训练配置的对象
    """
    # 步骤1: 设置环境变量和初始化Ray
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    
    # 检查是否启用本地调试模式
    debug_locally = config.get("debug_locally", False)
    
    if not debug_locally:
        if not ray.is_initialized():
            # 初始化本地Ray集群
            ray.init(
                runtime_env={
                    "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}
                },
                num_cpus=config.ray_init.num_cpus,
            )

    # 检查是否是快速测试模式
    quick_test_mode = config.get("quick_test_mode", False)
    if quick_test_mode:
        print("\n===== 快速测试模式 =====")
        print("仅初始化环境，不执行训练流程")
        print("Ray集群已初始化")
        print("配置检查完成")
        print("=========================\n")
        return

    # 步骤2: 创建任务运行器并执行
    if debug_locally:
        # 本地调试模式：直接在主进程中执行，便于设置断点
        print("\n===== 本地调试模式 =====")
        print("在主进程中直接执行TaskRunner")
        print("断点应该可以正常工作")
        print("=========================\n")
        runner = LocalTaskRunner()
        runner.run(config)  # 直接调用，不使用asyncio.run
    else:
        # 正常Ray远程执行模式
        runner = TaskRunner.remote()
        ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # 确保主任务不在头节点上调度
class TaskRunner:
    """Ray远程任务运行器，负责执行PPO训练的实际任务"""
    
    def run(self, config):
        """
        运行PPO训练任务
        
        参数:
            config: 包含训练配置的对象
        """
        # 步骤1: 打印初始配置
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True会计算符号值
        OmegaConf.resolve(config)

        # 步骤2: 从HDFS下载检查点到本地
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # 步骤3: 实例化tokenizer和processor
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)  # 用于多模态LLM，可能为None

        # 步骤4: 根据策略定义工作器类
        if config.actor_rollout_ref.actor.strategy == "fsdp":
            # 使用FSDP (Fully Sharded Data Parallel) 策略
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            # 根据rollout模式选择actor类
            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            # 使用Megatron策略
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        # 步骤5: 设置资源池和角色映射
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # 步骤6: 设置奖励模型（如果启用）
        if config.reward_model.enable:
            if config.reward_model.strategy == "fsdp":
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # 步骤7: 设置参考模型（如果需要）
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # 步骤8: 加载奖励函数
        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        if reward_manager_name == 'naive':
            from verl.workers.reward_manager import NaiveRewardManager
            reward_manager_cls = NaiveRewardManager
        elif reward_manager_name == 'prime':
            from verl.workers.reward_manager import PrimeRewardManager
            reward_manager_cls = PrimeRewardManager
        else:
            raise NotImplementedError
        
        compute_score = compute_kg_extraction_reward
        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1,          
                                       compute_score=compute_score)

        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, 
                                           compute_score=compute_score)
        
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        
        # 步骤9: 创建PPO训练器并开始训练
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            collate_fn=collate_fn,
        )
        trainer.init_workers()
        trainer.fit()


# 本地调试版本的TaskRunner，无Ray装饰器
class LocalTaskRunner:
    """本地任务运行器，用于调试时在主进程中执行，支持断点调试"""
    
    def run(self, config):
        """
        运行PPO训练任务
        
        参数:
            config: 包含训练配置的对象
        """
        # 步骤1: 打印初始配置
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True会计算符号值
        OmegaConf.resolve(config)

        # 步骤2: 从HDFS下载检查点到本地
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        print(f"模型本地路径: {local_path}")

        # 步骤3: 实例化tokenizer和processor
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)  # 用于多模态LLM，可能为None

        # 为简化调试，创建一个简单的调试点
        print("\n===== 开始调试会话 =====")
        print("您现在可以在此处设置断点，检查关键对象")
        print("对象可用性:")
        print(f"- config: {'可用' if config else '不可用'}")
        print(f"- tokenizer: {'可用' if tokenizer else '不可用'}")
        print(f"- processor: {'可用' if processor else '不可用'}")
        
        # 加载奖励函数
        print("\n加载计算奖励函数...")
        compute_score = compute_kg_extraction_reward
        
        
        # 创建一个简单的测试数据用于调试
        print("\n创建测试数据...")
        test_data = {
            "prompt": "请从以下文本中提取知识三元组: 苹果公司总部位于加利福尼亚州的库比蒂诺。",
            "response": """<think>
分析文本，我们可以确定以下信息：
1. 苹果公司是一家公司
2. 库比蒂诺是苹果公司的总部所在地
3. 加利福尼亚州是库比蒂诺所在的州
</think>
<answer>
{
    "nodes": [
        {"id": "1", "name": "苹果公司", "type": "Company"},
        {"id": "2", "name": "库比蒂诺", "type": "Location"},
        {"id": "3", "name": "加利福尼亚州", "type": "Location"}
    ],
    "edges": [
        {"from": "1", "to": "2", "label": "总部位于"},
        {"from": "2", "to": "3", "label": "位于"}
    ]
}
</answer>
"""
        }
        
        
        # 按照compute_kg_extraction_reward函数的要求提供参数
        data_source = "test_data"
        solution_str = test_data["response"]
        ground_truth = {
            "ground_truth_answer": """
{
    "nodes": [
        {"id": "1", "name": "苹果公司", "type": "Company"},
        {"id": "2", "name": "库比蒂诺", "type": "Location"},
        {"id": "3", "name": "加利福尼亚州", "type": "Location"}
    ],
    "edges": [
        {"from": "1", "to": "2", "label": "总部位于"},
        {"from": "2", "to": "3", "label": "位于"}
    ]
}
""",
            "ground_truth_reasoning": "这段文本描述了苹果公司的总部位置，其中包含两个关系：苹果公司的总部位于库比蒂诺，以及库比蒂诺位于加利福尼亚州。"
        }
        
        reward = compute_score(data_source, solution_str, ground_truth)
        print(f"计算的奖励值: {reward}")
        
        # 简化的退出
        print("\n调试完成。您可以在此处添加更多断点和调试代码。")
        print("=========================\n")
        
        # 如果需要继续训练流程，取消下面的注释
        # from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        # trainer = RayPPOTrainer(...)
        # trainer.init_workers()
        # trainer.fit()  # 移除await关键字
        
        return


def print_debug_info(config, args):
    """打印调试信息"""
    print("\n" + "="*50)
    print("调试信息：")
    print(f"配置文件默认路径: {args.config_path}")
    print(f"模型默认路径: {args.model_path}")
    print(f"输出目录默认值: {args.output_dir}")
    
    print("\n实际配置值：")
    try:
        print(f"训练数据路径: {config.data.train_path}")
        print(f"模型路径: {config.actor_rollout_ref.model.path}")
        print(f"输出目录: {config.trainer.output_dir}")
    except Exception as e:
        print(f"获取实际配置值失败: {e}")
    
    print("\nHydra命令行参数:")
    for arg in sys.argv[1:]:
        print(f"  {arg}")
    print("="*50 + "\n")

def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset.

    Arguments:
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset

def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # use sampler for better ckpt resume
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler

if __name__ == "__main__":
    # 直接调用main()函数，由Hydra处理命令行参数
    
    main()
