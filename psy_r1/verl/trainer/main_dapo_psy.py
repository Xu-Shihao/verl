#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAPO训练器 - 心理诊断任务专用版本
基于原始main_dapo.py，集成心理诊断专用奖励函数
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import is_cuda_available

# 导入DAPO Ray训练器
from psy_r1.verl.trainer.dapo_ray_trainer import RayDAPOTrainer

# 导入心理诊断奖励函数
from psy_r1.verl.utils.dapo_reward_score_psy import register_psy_reward_function, create_dapo_psy_reward_fn


@hydra.main(config_path="./config", config_name="dapo_trainer_psy", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={
                "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}
            },
            num_cpus=config.ray_init.num_cpus,
        )

    if (
        is_cuda_available
        and OmegaConf.select(config.trainer, "profile_steps") is not None
        and len(OmegaConf.select(config.trainer, "profile_steps")) > 0
    ):
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        print("=== 心理诊断DAPO任务配置 ===")

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
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

        # reward model setup
        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # 创建专门的心理诊断奖励函数（使用新的DAPO专用函数）
        print("=== 初始化DAPO心理诊断奖励函数 ===")
        
        # 检查日志配置
        show_train_examples = config.reward_model.get("show_training_examples", True)
        show_val_examples = config.reward_model.get("show_val_examples", True)
        symptom_alpha = config.reward_model.get("symptom_alpha", 0.1)
        
        # 检查症状奖励配置
        use_symptom_reward = config.reward_model.get("use_symptom_reward", False)
        
        print("是否开启症状奖励: ", use_symptom_reward)
        
        # 根据配置确定日志模式
        train_log_mode = "train" if show_train_examples else None
        val_log_mode = "val" if show_val_examples else None
        
        # 创建训练和验证的奖励函数
        reward_fn = create_dapo_psy_reward_fn(
            is_validation=train_log_mode, 
            tokenizer=tokenizer, 
            use_symptom_reward=use_symptom_reward, 
            symptom_alpha=symptom_alpha
        )
        val_reward_fn = create_dapo_psy_reward_fn(
            is_validation=val_log_mode, 
            tokenizer=tokenizer, 
            use_symptom_reward=use_symptom_reward, 
            symptom_alpha=symptom_alpha
        )
        
        # 打印配置状态
        train_status = "开启日志" if show_train_examples else "关闭日志"
        val_status = "开启日志" if show_val_examples else "关闭日志"
        symptom_status = "启用症状奖励" if use_symptom_reward else "仅诊断奖励"
        print(f"DAPO奖励函数配置 - 训练模式: {train_status}, 验证模式: {val_status}, 奖励模式: {symptom_status}")
        print(f"症状奖励权重: {symptom_alpha}")
        
        print("=== DAPO奖励函数初始化完成 ===")
        
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = RayDAPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
